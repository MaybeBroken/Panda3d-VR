"""
Panda3D <-> OpenXR integration.

Frame pipeline (all on Panda's main thread, one GL context, zero CPU copies):

  1. ``vr-frame`` task (sort -1000): poll XR events, xrWaitFrame (this is what
     paces the app to the headset refresh rate), xrBeginFrame, locate eyes,
     head, controllers, hands and gaze for the *predicted display time*.
  2. User tasks run with those fresh poses.
  3. Panda renders each eye into an offscreen FBO with its exact off-axis
     frustum (MSAA resolved by Panda).
  4. A draw callback on a tiny "submit" buffer runs inside Panda's GL context:
     it copies each eye texture into the OpenXR swapchain image on the GPU
     (glCopyImageSubData) and calls xrEndFrame.

The old implementation read both eyes back to system RAM every frame, resized
them with OpenCV and re-uploaded them from a second GL context on another
thread; that round trip is gone entirely.
"""

import atexit
import builtins
import logging
import sys
import time
from ctypes import c_void_p, cast, pointer

from panda3d.core import (
    BitMask32,
    Camera,
    CardMaker,
    ClockObject,
    ConfigVariableBool,
    ConfigVariableString,
    CullBinManager,
    FrameBufferProperties,
    GraphicsOutput,
    GraphicsPipe,
    NodePath,
    PandaNode,
    PerspectiveLens,
    PythonCallbackObject,
    Texture,
    TextNode,
    WindowProperties,
    loadPrcFileData,
)
from direct.showbase.DirectObject import DirectObject
from direct.showbase.MessengerGlobal import messenger
from direct.showbase.ShowBase import ShowBase

from .controller import Controller
from .simulator import DesktopSimulator

log = logging.getLogger("panda3d_vr")

try:
    import xr

    from . import _gl
    from .hands import HandTracking
    from .input import XrInput
    from .layers import Passthrough, QuadLayer
    from .runtime import XrCallError, XrRuntime
    from .xrmath import apply_pose, fov_to_film

    XR_AVAILABLE = sys.platform == "win32"
    _ALPHA_LAYER_FLAGS = (xr.CompositionLayerFlags.BLEND_TEXTURE_SOURCE_ALPHA_BIT
                          | xr.CompositionLayerFlags.UNPREMULTIPLIED_ALPHA_BIT).value
    _XR_IMPORT_ERROR = None if XR_AVAILABLE else "only Windows is supported for now"
except Exception as _e:  # pragma: no cover - missing pyopenxr / PyOpenGL
    XR_AVAILABLE = False
    _XR_IMPORT_ERROR = _e

__all__ = ["VRManager", "BaseVrApp", "XR_AVAILABLE", "EYE_MASKS"]

# Camera-mask bits reserved for the eyes; used to keep each eye's
# hidden-area mesh out of the other eye (and out of the desktop camera).
EYE_MASKS = (BitMask32.bit(28), BitMask32.bit(29))

MIRROR_MODES = ("left", "right", "both", "spectator", "none")

_MASK_BIN = "vr-hidden-area"


class _Eye:
    __slots__ = (
        "index", "cam", "lens", "buffer", "region", "color_tex", "depth_tex",
        "swapchain", "copier", "depth_swapchain", "depth_copier", "mask",
        "fov_key", "active",
    )

    def __init__(self, index):
        self.index = index
        for s in self.__slots__[1:]:
            setattr(self, s, None)
        self.active = True


class VRManager(DirectObject):
    """
    Adds OpenXR rendering, tracking and input to an existing ShowBase.

    Scene graph::

        render
          rig                  <- move/rotate this for locomotion (``player``)
            tracking_space     <- recentering offset lives here
              head, eye cameras, left/right grip+aim, hand joints, gaze

    Parameters (all optional):
        app_name          Name reported to the runtime.
        world_scale       Panda units per metre (1.0 = metres).
        near, far         Clip planes in metres.
        resolution_scale  Multiplier on the runtime's recommended eye size.
        msaa              Multisamples for eye buffers (0 to disable).
        tracking          "local_floor" (default), "stage" or "local".
        mirror            Desktop window: "left", "right", "both",
                          "spectator" (normal Panda camera on the head), "none".
        submit_depth      Send depth to the compositor (sharper reprojection /
                          ASW) when the runtime supports it.
        visibility_mask   Stencil out pixels hidden by the lenses (free GPU time).
        hand_tracking     Articulated hand tracking when supported.
        eye_tracking      Eye gaze pose when supported.
        passthrough       Start with Meta passthrough behind the scene.
        blend_mode        "opaque", "additive" or "alpha_blend" (AR headsets).
        refresh_rate      Request a display refresh rate (Hz) when supported.
        fallback          "simulator" (keyboard/mouse) or None when no headset.
        exit_on_session_end  Quit the app when the user quits from the headset.
        show_controllers  Draw simple controller models and pointer rays.
        show_stats        On-screen frame timing overlay.
        debug             Enable XR_EXT_debug_utils + verbose logging.
        custom_actions    {name: {"kind": "bool"|"float"|"vec2"|"pose",
                          "bindings": {profile: path}}}
        extensions        Extra OpenXR extension names to enable.
    """

    def __init__(
        self,
        base=None,
        *,
        app_name="Panda3D VR",
        world_scale=1.0,
        near=0.05,
        far=1000.0,
        resolution_scale=1.0,
        msaa=4,
        tracking="local_floor",
        mirror="left",
        submit_depth=True,
        visibility_mask=True,
        hand_tracking=True,
        eye_tracking=True,
        passthrough=False,
        blend_mode="opaque",
        refresh_rate=None,
        fallback="simulator",
        exit_on_session_end=True,
        show_controllers=True,
        show_stats=False,
        debug=False,
        custom_actions=None,
        extensions=(),
        reconnect_interval=2.0,
        eye_height=1.65,
        enabled=True,
    ):
        DirectObject.__init__(self)
        self.base = base or builtins.base
        if self.base.win is None:
            raise RuntimeError("VRManager needs a ShowBase with a window")
        self.render = self.base.render
        self.app_name = app_name
        self.world_scale = float(world_scale)
        self.near = near
        self.far = far
        self.resolution_scale = resolution_scale
        self.msaa = msaa
        self.tracking = tracking
        self.submit_depth = bool(submit_depth)
        self.use_visibility_mask = visibility_mask
        self.want_hand_tracking = hand_tracking
        self.want_eye_tracking = eye_tracking
        self.want_passthrough = passthrough
        self.blend_mode = blend_mode
        self.requested_refresh_rate = refresh_rate
        self.exit_on_session_end = exit_on_session_end
        self.debug = debug
        self.custom_actions = dict(custom_actions or {})
        self.extensions = tuple(extensions)
        self.reconnect_interval = reconnect_interval
        self.eye_height = eye_height
        if debug:
            logging.basicConfig()
            log.setLevel(logging.DEBUG)

        # --- scene graph -------------------------------------------------
        self.rig = self.render.attach_new_node("vr-rig")
        self.tracking_space = self.rig.attach_new_node("vr-tracking-space")
        self.head = self.tracking_space.attach_new_node("vr-head")
        self.gaze = self.tracking_space.attach_new_node("vr-gaze")
        self.left = Controller(self, "left", self.tracking_space)
        self.right = Controller(self, "right", self.tracking_space)
        self.controllers = (self.left, self.right)
        self.hands = None
        self.head_tracked = False
        self.gaze_valid = False

        self._eyes = [_Eye(0), _Eye(1)]
        for eye in self._eyes:
            lens = PerspectiveLens()
            lens.set_near_far(near * self.world_scale, far * self.world_scale)
            cam = Camera("vr-eye-%s" % ("left", "right")[eye.index], lens)
            cam.set_camera_mask(PandaNode.get_all_camera_mask() & ~EYE_MASKS[1 - eye.index])
            eye.cam = self.tracking_space.attach_new_node(cam)
            eye.lens = lens
        self.eye_cameras = [e.cam for e in self._eyes]

        # The desktop camera follows the head (spectator / simulator view).
        self.base.camera.reparent_to(self.head)
        self.base.camera.set_pos_hpr(0, 0, 0, 0, 0, 0)
        self.base.camLens.set_near_far(near * self.world_scale, far * self.world_scale)

        # --- state ---------------------------------------------------------
        self.rt = None
        self.input = None
        self.passthrough = None
        self.status = "disabled" if not enabled else "disconnected"
        self._enabled = enabled
        self._quad_layers = []
        self._pending_quads = []
        self._deferred = []  # (event name | callable, args) raised from the GL callback
        self._session_requested = False
        self._frame_open = False
        self._render_frame = False
        self._teardown = None
        self._next_connect = 0.0
        self._waiting_logged = False
        self._submit_buffer = None
        self._gsg = self.base.win.get_gsg()
        self._hdc = self._hglrc = None
        self._proj_layer = None
        self._mask_dirty = False
        self._mask_data = [None, None]
        self._clear_color = None
        self._clock_mode = None
        self._floor_offset = 0.0
        self._shutdown_done = False
        self.stats = {
            "fps": 0.0, "xr_wait_ms": 0.0, "refresh_rate": 0.0, "frames": 0,
            "skipped": 0, "eye_size": (0, 0),
        }

        CullBinManager.get_global_ptr().add_bin(_MASK_BIN, CullBinManager.BT_fixed, -100)

        self.simulator = DesktopSimulator(self, eye_height=eye_height) if fallback == "simulator" else None
        self.mirror = None
        self._mirror_cards = []
        self._controller_models = []
        self._stats_text = None
        self.set_mirror(mirror)
        if show_controllers:
            self.show_controller_models()
        if show_stats:
            self.set_stats_visible(True)

        if ConfigVariableBool("sync-video", True).value:
            log.warning(
                "sync-video is on: the desktop window's vsync will cap the headset "
                "frame rate.  Put 'sync-video false' in your PRC config (BaseVrApp "
                "does this automatically).")
        if ConfigVariableString("threading-model", "").value:
            log.warning("threading-model is set; panda3d_vr is designed for the "
                        "single-threaded pipeline and may stutter.")

        self.base.taskMgr.add(self._frame_task, "vr-frame", sort=-1000)
        self.accept("window-event", self._on_window_event)
        prev_exit = self.base.exitFunc

        def _exit_hook():
            self.shutdown()
            if prev_exit:
                prev_exit()

        self.base.exitFunc = _exit_hook
        atexit.register(self.shutdown)

        if enabled and not XR_AVAILABLE:
            log.warning("OpenXR unavailable (%s); running without a headset", _XR_IMPORT_ERROR)
        # Simulate until a headset shows up (hot-plug switches over automatically).
        self._set_simulated(True)

    # ================================================================ public

    @property
    def player(self):
        return self.rig

    @property
    def connected(self):
        return self.rt is not None and self.rt.system_id is not None

    @property
    def running(self):
        return self.connected and self.rt.running

    @property
    def focused(self):
        return self.running and self.rt.focused

    @property
    def simulated(self):
        return self.simulator is not None and self.simulator.enabled

    @property
    def system_name(self):
        return self.rt.system_name if self.connected else None

    def recenter(self):
        """Yaw-only recenter: the current head position/heading becomes the origin."""
        ts = self.tracking_space
        head_pos = self.head.get_pos()
        h = self.head.get_h()
        ts.set_pos(0, 0, self._floor_offset)
        ts.set_hpr(-h, 0, 0)
        p = self.rig.get_relative_point(ts, head_pos)
        ts.set_pos(-p.x, -p.y, self._floor_offset)
        messenger.send("vr-recentered", [self])

    def set_world_scale(self, scale):
        self.world_scale = float(scale)
        for eye in self._eyes:
            eye.lens.set_near_far(self.near * scale, self.far * scale)
            eye.fov_key = None
        self._mask_dirty = True

    def set_clip_planes(self, near, far):
        self.near, self.far = near, far
        self.set_world_scale(self.world_scale)

    def vibrate(self, hand, amplitude=0.5, duration=0.05, frequency=0.0):
        self.controllers[hand if isinstance(hand, int) else ("left", "right").index(hand)].vibrate(
            amplitude, duration, frequency)

    def set_tracking_space(self, kind):
        """Switch between "local_floor", "stage" and "local" at runtime."""
        self.tracking = kind
        if self.rt is not None and self.rt.session is not None:
            actual = self.rt.set_tracking_space(kind)
            self._apply_floor_offset(actual)
            return actual
        return None

    def get_play_area(self):
        """(width, depth) of the guardian rectangle in metres, or None."""
        return self.rt.play_area() if self.rt is not None else None

    def get_refresh_rates(self):
        return self.rt.refresh_rates() if self.rt is not None else []

    def get_refresh_rate(self):
        return self.rt.refresh_rate() if self.rt is not None else None

    def set_refresh_rate(self, hz):
        self.requested_refresh_rate = hz
        if self.rt is not None and self.rt.session is not None:
            return self.rt.request_refresh_rate(hz)
        return False

    def set_performance_level(self, cpu=None, gpu=None):
        """Levels: "power_savings", "sustained_low", "sustained_high", "boost"."""
        if self.rt is None:
            return False
        ok = True
        if cpu:
            ok &= self.rt.set_performance_level("cpu", cpu)
        if gpu:
            ok &= self.rt.set_performance_level("gpu", gpu)
        return ok

    def set_passthrough(self, enabled):
        self.want_passthrough = enabled
        if self.passthrough is not None:
            try:
                self.passthrough.set_enabled(enabled)
            except Exception as e:
                log.warning("Passthrough toggle failed: %s", e)
        self._clear_color = None

    @property
    def passthrough_active(self):
        return self.want_passthrough and self.passthrough is not None and self.passthrough.handle is not None

    def create_quad_layer(self, resolution=(1024, 512), size=(1.0, 0.5), parent=None,
                          name=None, eye="both"):
        """Create a crisp compositor-rendered UI panel.  See ``layers.QuadLayer``."""
        if not XR_AVAILABLE:
            raise RuntimeError("Quad layers need OpenXR")
        q = QuadLayer(self, resolution, size, parent, name or "quad-layer-%d" % len(self._quad_layers), eye)
        self._quad_layers.append(q)
        self._pending_quads.append(q)
        return q

    def _remove_quad_layer(self, q):
        if q in self._quad_layers:
            self._quad_layers.remove(q)

    def request_exit(self):
        """Ask the runtime to end the session gracefully (then the app exits)."""
        if self.rt is not None:
            self.rt.request_exit()
        else:
            self.base.userExit()

    def get_eye_texture(self, eye=0):
        return self._eyes[eye].color_tex

    # ---------------------------------------------------------------- mirror

    def set_mirror(self, mode):
        if mode not in MIRROR_MODES:
            raise ValueError("mirror must be one of %s" % (MIRROR_MODES,))
        self.mirror = mode
        self._update_mirror()

    def cycle_mirror(self):
        self.set_mirror(MIRROR_MODES[(MIRROR_MODES.index(self.mirror) + 1) % len(MIRROR_MODES)])

    def _update_mirror(self):
        have_eyes = self._eyes[0].color_tex is not None and self.connected
        desktop = self.simulated or not have_eyes or self.mirror == "spectator"
        self.base.camNode.set_active(desktop)
        for card in self._mirror_cards:
            card.remove_node()
        self._mirror_cards = []
        if desktop or self.mirror == "none":
            return
        eyes = (0, 1) if self.mirror == "both" else ((1,) if self.mirror == "right" else (0,))
        for slot, i in enumerate(eyes):
            cm = CardMaker("vr-mirror-%d" % i)
            cm.set_frame(-1, 1, -1, 1)
            card = self.base.render2d.attach_new_node(cm.generate())
            card.set_texture(self._eyes[i].color_tex, 1)
            card.set_bin("background", -100)
            card.set_depth_write(False)
            card.set_depth_test(False)
            card.set_python_tag("slot", (slot, len(eyes)))
            self._mirror_cards.append(card)
        self._layout_mirror()

    def _layout_mirror(self):
        if not self._mirror_cards or self.base.win is None:
            return
        w = max(1, self.base.win.get_x_size())
        h = max(1, self.base.win.get_y_size())
        ew, eh = self.stats["eye_size"]
        if not ew:
            return
        n = len(self._mirror_cards)
        content = n * ew / float(eh)
        win = w / float(h)
        sx, sz = (content / win, 1.0) if win > content else (1.0, win / content)
        for card in self._mirror_cards:
            slot, count = card.get_python_tag("slot")
            card.set_scale(sx / count, 1, sz)
            card.set_x((-1 + (2 * slot + 1) / count) * sx)

    def _on_window_event(self, win):
        if win == self.base.win:
            self._layout_mirror()

    # ------------------------------------------------------ visual helpers

    def show_controller_models(self, show=True):
        from .geometry import box, line_node

        if not show:
            for m in self._controller_models:
                m.remove_node()
            self._controller_models = []
            return
        if self._controller_models:
            return
        for c, color in zip(self.controllers, ((0.3, 0.6, 1, 1), (1, 0.45, 0.3, 1))):
            body = c.grip.attach_new_node(box((0.02, 0.06, 0.02), (0, -0.02, 0), color, "vr-model"))
            body.set_scale(self.world_scale)
            ray = c.aim.attach_new_node(line_node([(0, 0, 0), (0, 1, 0)], (1, 1, 1, 0.6), "vr-ray"))
            ray.set_scale(1, 2.0 * self.world_scale, 1)
            ray.set_light_off(1)
            ray.set_transparency(True)
            c.model, c.ray = body, ray
            self._controller_models += [body, ray]

    def set_stats_visible(self, visible):
        from direct.gui.OnscreenText import OnscreenText

        if visible and self._stats_text is None:
            self._stats_text = OnscreenText(
                text="", parent=self.base.a2dTopLeft, pos=(0.05, -0.08), scale=0.045,
                align=TextNode.A_left, fg=(1, 1, 0.6, 1), shadow=(0, 0, 0, 1), mayChange=True)
            self._stats_next = 0.0
        elif not visible and self._stats_text is not None:
            self._stats_text.destroy()
            self._stats_text = None

    def enable_debug_keys(self):
        """r: recenter, v: cycle mirror mode, f3: stats overlay, p: passthrough."""
        self.accept("r", self.recenter)
        self.accept("v", self.cycle_mirror)
        self.accept("f3", lambda: self.set_stats_visible(self._stats_text is None))
        self.accept("p", lambda: self.set_passthrough(not self.want_passthrough))

    # ============================================================ internals

    def _set_simulated(self, on):
        if self.simulator is None:
            return
        if on:
            self.simulator.enable()
            self._set_clock(limited=90)
        else:
            self.simulator.disable()
        self._update_mirror()

    def _set_clock(self, limited=None):
        mode = limited or 0
        if mode == self._clock_mode:
            return
        self._clock_mode = mode
        clock = ClockObject.get_global_clock()
        if limited:
            clock.set_mode(ClockObject.M_limited)
            clock.set_frame_rate(limited)
        else:
            clock.set_mode(ClockObject.M_normal)

    def _haptic(self, hand, amplitude, duration, frequency):
        if self.input is not None and self.focused:
            self.input.haptic(hand, amplitude, duration, frequency)

    def _stop_haptic(self, hand):
        if self.input is not None and self.focused:
            self.input.stop_haptic(hand)

    # ------------------------------------------------------------ connection

    def _on_xr_event(self, name, **data):
        if name == "session-state":
            state = data["state"]
            messenger.send("vr-session-state", [state.name.lower()])
            if state == xr.SessionState.FOCUSED:
                messenger.send("vr-focus-gained")
            elif state == xr.SessionState.VISIBLE:
                messenger.send("vr-focus-lost")
            elif state == xr.SessionState.READY:
                self._on_session_ready()
            elif state == xr.SessionState.EXITING:
                messenger.send("vr-session-exit")
                if self.exit_on_session_end:
                    self.base.taskMgr.do_method_later(0, lambda _task: self.base.userExit(), "vr-exit")
                else:
                    self._teardown = "session"
            elif state == xr.SessionState.LOSS_PENDING:
                log.warning("Headset lost; will try to reconnect")
                self._teardown = "instance"
        elif name == "instance-lost":
            self._teardown = "instance"
        elif name == "interaction-profile-changed":
            if self.input is not None:
                self.input.refresh_profiles()
            for c in self.controllers:
                messenger.send("vr-controller-profile", [c])
        elif name == "reference-space-changed":
            messenger.send("vr-recentered", [self])
        elif name == "visibility-mask-changed":
            self._mask_data[data["view_index"] % 2] = None
            self._mask_dirty = True
        elif name == "refresh-rate-changed":
            messenger.send("vr-refresh-rate-changed", [data["new"]])
        elif name == "user-presence":
            messenger.send("vr-user-presence", [data["present"]])
        elif name == "performance-notification":
            messenger.send("vr-performance-notification", [data])

    def _try_connect(self):
        now = time.perf_counter()
        self._next_connect = now + self.reconnect_interval
        try:
            if self.rt is None:
                features = {"depth", "visibility_mask", "refresh_rate", "performance",
                            "local_floor", "user_presence", "hand_interaction", "debug"}
                if self.want_hand_tracking:
                    features.add("hand_tracking")
                if self.want_eye_tracking:
                    features.add("eye_gaze")
                # Always requested so passthrough can be toggled on later.
                features.add("passthrough")
                self.rt = XrRuntime(self.app_name, features, self.extensions, self.debug,
                                    self.blend_mode, self._on_xr_event)
                self.rt.create_instance()
            if self.rt.try_get_system():
                self._waiting_logged = False
                self._on_system_ready()
            else:
                self.status = "waiting"
                if not self._waiting_logged:
                    log.info("OpenXR runtime found (%s); waiting for a headset...", self.rt.runtime_name)
                    self._waiting_logged = True
        except Exception as e:
            if not self._waiting_logged:
                log.warning("OpenXR not available yet: %s", e)
                self._waiting_logged = True
            if self.rt is not None:
                self.rt.destroy()
            self.rt = None
            self.status = "disconnected"

    def _on_system_ready(self):
        rt = self.rt
        vc = rt.view_configs[0]
        w = int(min(vc.max_image_rect_width, round(vc.recommended_image_rect_width * self.resolution_scale)))
        h = int(min(vc.max_image_rect_height, round(vc.recommended_image_rect_height * self.resolution_scale)))
        if self.stats["eye_size"] != (w, h) or self._eyes[0].buffer is None:
            self._create_eye_buffers(w, h)
        if self._submit_buffer is None:
            self._create_submit_buffer()
        self._session_requested = True
        self.status = "connected"
        self._set_simulated(False)
        messenger.send("vr-connected", [self])

    def _make_buffer(self, name, w, h, sort, msaa, color_tex, depth_tex=None, float_depth=False):
        fb = FrameBufferProperties()
        fb.set_rgba_bits(8, 8, 8, 8)
        fb.set_depth_bits(32 if float_depth else 24)
        fb.set_float_depth(float_depth)
        if msaa > 1:
            fb.set_multisamples(msaa)
        base = self.base
        buf = base.graphicsEngine.make_output(
            base.pipe, name, sort, fb, WindowProperties.size(w, h),
            GraphicsPipe.BF_refuse_window, base.win.get_gsg(), base.win)
        if buf is None:
            raise RuntimeError("Could not create %dx%d offscreen buffer %r" % (w, h, name))
        buf.add_render_texture(color_tex, GraphicsOutput.RTM_bind_or_copy, GraphicsOutput.RTP_color)
        if depth_tex is not None:
            buf.add_render_texture(depth_tex, GraphicsOutput.RTM_bind_or_copy, GraphicsOutput.RTP_depth)
        return buf

    def _create_eye_buffers(self, w, h):
        engine = self.base.graphicsEngine
        for eye in self._eyes:
            if eye.buffer is not None:
                engine.remove_window(eye.buffer)
            eye.color_tex = Texture("vr-eye-%d" % eye.index)
            eye.color_tex.set_minfilter(Texture.FT_linear)
            eye.color_tex.set_magfilter(Texture.FT_linear)
            eye.depth_tex = Texture("vr-eye-depth-%d" % eye.index) if self.submit_depth else None
            eye.buffer = self._make_buffer(
                "vr-eye-%d" % eye.index, w, h, -30 + eye.index, self.msaa,
                eye.color_tex, eye.depth_tex, float_depth=self.submit_depth)
            eye.buffer.set_clear_color_active(True)
            eye.buffer.set_clear_depth_active(True)
            eye.region = eye.buffer.make_display_region()
            eye.region.set_camera(eye.cam)
            eye.active = True
        self._clear_color = None
        self.stats["eye_size"] = (w, h)
        log.info("Eye buffers: %dx%d, %sx MSAA, depth submission %s",
                 w, h, self.msaa or "no", "on" if self.submit_depth else "off")
        self._update_mirror()

    def _create_submit_buffer(self):
        tex = Texture("vr-submit")
        buf = self._make_buffer("vr-submit", 16, 16, -10, 0, tex)
        buf.set_clear_color_active(False)
        buf.set_clear_depth_active(False)
        cam = Camera("vr-submit-cam")
        cam.set_scene(NodePath("vr-submit-scene"))
        cam_np = NodePath(cam)
        dr = buf.make_display_region()
        dr.set_camera(cam_np)
        dr.set_draw_callback(PythonCallbackObject(self._draw_callback))
        self._submit_buffer = buf
        self._submit_cam = cam_np

    def _set_eyes_active(self, active):
        for eye in self._eyes:
            if eye.buffer is not None and eye.active != active:
                eye.buffer.set_active(active)
                eye.active = active

    # --------------------------------------------------- GL-side (callback)

    def _native_id(self, tex):
        gsg = self._gsg
        return tex.prepare_now(0, gsg.get_prepared_objects(), gsg).get_native_id()

    def _draw_callback(self, _cbdata):
        try:
            if self._teardown:
                self._do_teardown()
                return
            if self._session_requested and self.rt is not None and self.rt.session is None:
                self._session_requested = False
                self._create_session()
                return
            if self._pending_quads and self.rt is not None and self.rt.session is not None:
                for q in self._pending_quads:
                    if q in self._quad_layers:
                        q._create(self.rt, self._native_id)
                self._pending_quads = []
            if self._frame_open:
                self._frame_open = False
                self._submit_frame()
        except Exception as e:
            self._frame_open = False
            log.exception("VR submit failed: %s", e)
            self._teardown = "instance"

    def _create_session(self):
        rt = self.rt
        self._hdc, self._hglrc = _gl.current_context()
        log.debug("GL %s", _gl.gl_version())
        rt.create_session(self._hdc, self._hglrc, self.tracking)
        _gl.ensure_current(self._hdc, self._hglrc)
        self._apply_floor_offset(rt.space_type)

        # Swapchains sized/formatted to match Panda's eye textures exactly so the
        # per-frame transfer is a single raw GPU copy.
        n = len(rt.view_configs)
        self._proj_views = (xr.CompositionLayerProjectionView * n)(
            *[xr.CompositionLayerProjectionView() for _ in range(n)])
        self._depth_infos = []
        w, h = self.stats["eye_size"]
        depth_ok = self.submit_depth and rt.has("depth")
        for i, eye in enumerate(self._eyes):
            src = self._native_id(eye.color_tex)
            src_fmt = _gl.texture_internal_format(src)
            fmt = rt.pick_color_format(_gl.COLOR_FORMAT_PREFERENCE)
            eye.swapchain = rt.create_swapchain(fmt, w, h)
            eye.copier = _gl.TextureCopier(src_fmt, fmt)
            eye.swapchain.sub_image(self._proj_views[i].sub_image)
            if depth_ok and eye.depth_tex is not None:
                dfmt = _gl.texture_internal_format(self._native_id(eye.depth_tex))
                if dfmt in rt.swapchain_formats:
                    eye.depth_swapchain = rt.create_swapchain(dfmt, w, h, depth=True)
                    eye.depth_copier = _gl.TextureCopier(dfmt, dfmt, depth=True)
                    info = xr.CompositionLayerDepthInfoKHR(
                        min_depth=0.0, max_depth=1.0, near_z=self.near, far_z=self.far)
                    eye.depth_swapchain.sub_image(info.sub_image)
                    self._depth_infos.append(info)
                    self._proj_views[i].next = cast(pointer(info), c_void_p)
                else:
                    log.info("Depth format 0x%04X not accepted by the runtime; depth submission off", dfmt)
                    depth_ok = False
            log.info("Eye %d: swapchain %dx%d format 0x%04X (%s from 0x%04X)", i, w, h, fmt,
                     eye.copier.mode, src_fmt)
        self._proj_layer = xr.CompositionLayerProjection(space=rt.space, views=self._proj_views)
        self._proj_ptr = rt.layer_pointer(self._proj_layer)
        self._head_loc = xr.SpaceLocation()

        self.input = XrInput(rt, self.controllers, self.gaze, self.custom_actions)
        self.input.create()
        self.input.attach()

        if self.want_hand_tracking and rt.has("hand_tracking"):
            try:
                self.hands = HandTracking(rt, self.tracking_space)
                self.hands.create()
            except Exception as e:
                log.info("Hand tracking unavailable: %s", e)
                self.hands = None

        if rt.has("passthrough"):
            try:
                self.passthrough = Passthrough(rt, self.want_passthrough)
                self.passthrough.create()
            except Exception as e:
                log.info("Passthrough unavailable: %s", e)
                self.passthrough = None

        # Quad swapchains are made on the next callback, after the session exists.
        self._pending_quads = list(self._quad_layers)

        if self.use_visibility_mask and rt.has("visibility_mask"):
            self._mask_data = [None, None]
            self._mask_dirty = True
        if self.requested_refresh_rate:
            try:
                rt.request_refresh_rate(self.requested_refresh_rate)
            except Exception as e:
                log.warning("Refresh rate request failed: %s", e)
        self.status = "session"
        self._deferred.append(("vr-session-created", [self]))

    def _submit_frame(self):
        rt = self.rt
        fs = rt.frame_state
        layers = []
        if self._render_frame:
            native = self._native_id
            views = rt.views
            pviews = self._proj_views
            for i, eye in enumerate(self._eyes):
                sc = eye.swapchain
                dst = sc.acquire()
                eye.copier.copy(native(eye.color_tex), dst, sc.width, sc.height)
                sc.release()
                if eye.depth_swapchain is not None:
                    dsc = eye.depth_swapchain
                    ddst = dsc.acquire()
                    eye.depth_copier.copy(native(eye.depth_tex), ddst, dsc.width, dsc.height)
                    dsc.release()
                pviews[i].pose = views[i].pose
                pviews[i].fov = views[i].fov
            layer = self._proj_layer
            layer.space = rt.space
            if self.passthrough_active:
                layer.layer_flags = _ALPHA_LAYER_FLAGS
                layers.append(self.passthrough.ptr)
            else:
                layer.layer_flags = 0
            layers.append(self._proj_ptr)
            scale = self.world_scale
            for q in self._quad_layers:
                if q.visible and q.swapchain is not None and len(layers) < rt.max_layers:
                    layers.append(q._submit(rt, native, scale))
        rt.end_frame(fs.predicted_display_time, layers)
        _gl.ensure_current(self._hdc, self._hglrc)

    def _do_teardown(self):
        mode = self._teardown
        self._teardown = None
        self._frame_open = False
        self._destroy_session_objects(gl=True)
        if self.rt is not None:
            if mode == "instance":
                self.rt.destroy()
                self.rt = None
                self.status = "disconnected"
            else:
                self.rt.destroy_session()
                self.status = "connected"
        self._next_connect = time.perf_counter() + self.reconnect_interval
        self._set_eyes_active(False)
        self._deferred.append(("vr-disconnected", [self]))
        self._deferred.append((self._set_simulated, [True]))

    def _destroy_session_objects(self, gl=False):
        if self.input is not None:
            self.input.destroy()
            self.input = None
        if self.hands is not None:
            self.hands.destroy()
            self.hands = None
        if self.passthrough is not None:
            self.passthrough.destroy()
            self.passthrough = None
        for q in self._quad_layers:
            q._destroy_xr()
        for eye in self._eyes:
            for attr in ("swapchain", "depth_swapchain"):
                sc = getattr(eye, attr)
                if sc is not None:
                    sc.destroy()
                    setattr(eye, attr, None)
            for attr in ("copier", "depth_copier"):
                c = getattr(eye, attr)
                if c is not None and gl:
                    c.destroy()
                setattr(eye, attr, None)
            if eye.mask is not None:
                eye.mask.remove_node()
                eye.mask = None
        self._proj_layer = None

    def shutdown(self):
        """Release all OpenXR resources.  Called automatically on exit."""
        if self._shutdown_done:
            return
        self._shutdown_done = True
        try:
            self._destroy_session_objects(gl=False)
            if self.rt is not None:
                self.rt.destroy()
                self.rt = None
        except Exception as e:  # pragma: no cover
            log.debug("shutdown: %s", e)

    def destroy(self):
        """Shut down XR and remove everything VRManager added to the ShowBase."""
        self.shutdown()
        self.ignore_all()
        base = self.base
        base.taskMgr.remove("vr-frame")
        engine = base.graphicsEngine
        for q in list(self._quad_layers):
            q.destroy()
        for eye in self._eyes:
            if eye.buffer is not None:
                engine.remove_window(eye.buffer)
                eye.buffer = None
        if self._submit_buffer is not None:
            engine.remove_window(self._submit_buffer)
            self._submit_buffer = None
        self.set_stats_visible(False)
        for card in self._mirror_cards:
            card.remove_node()
        self._mirror_cards = []
        base.camera.wrt_reparent_to(self.render)
        base.camNode.set_active(True)
        self.rig.remove_node()
        self._set_clock(None)

    # ------------------------------------------------------------ app side

    def _on_session_ready(self):
        # Clock pacing is now done by xrWaitFrame.
        self._set_clock(None)

    def _apply_floor_offset(self, space_type):
        # A plain LOCAL space puts the origin at eye level; lift the rig so the
        # floor stays at z=0 in Panda.
        self._floor_offset = self.eye_height * self.world_scale if space_type == "local" else 0.0
        self.tracking_space.set_z(self._floor_offset)

    def _frame_task(self, task):
        if self._deferred:
            # Handlers run here, never inside the GL draw callback.
            pending, self._deferred = self._deferred, []
            for what, args in pending:
                if callable(what):
                    what(*args)
                else:
                    messenger.send(what, args)
        rt = self.rt
        dt = ClockObject.get_global_clock().get_dt()
        if not (self._enabled and XR_AVAILABLE) or rt is None or rt.system_id is None:
            if self._enabled and XR_AVAILABLE and time.perf_counter() >= self._next_connect:
                self._try_connect()
            if self.simulated:
                self.simulator.update(dt)
                self._update_stats(None, True, 0.0)
            return task.cont

        try:
            rt.poll_events()
        except XrCallError as e:
            log.warning("%s", e)
            self._teardown = "instance"
        if self._teardown or rt.session is None or not rt.running:
            if self._teardown is None and rt.session is None:
                self._session_requested = True
            self._set_eyes_active(False)
            if not self.simulated:
                self._set_clock(limited=30)
            for c in self.controllers:
                c._begin_frame()
            return task.cont

        self._set_clock(None)
        try:
            t0 = time.perf_counter()
            fs = rt.wait_and_begin_frame()
            wait_ms = (time.perf_counter() - t0) * 1000.0
            t = fs.predicted_display_time
            scale = self.world_scale
            render = bool(fs.should_render)
            if render:
                flags, views = rt.locate_views(t)
                render = bool(flags & xr.VIEW_STATE_ORIENTATION_VALID_BIT)
                if render:
                    for i, eye in enumerate(self._eyes):
                        v = views[i]
                        apply_pose(eye.cam, v.pose, scale)
                        f = v.fov
                        key = (f.angle_left, f.angle_right, f.angle_up, f.angle_down)
                        if key != eye.fov_key:
                            eye.fov_key = key
                            fw, fh, ox, oy = fov_to_film(f)
                            eye.lens.set_film_size(fw, fh)
                            eye.lens.set_film_offset(ox, oy)
                            eye.lens.set_focal_length(1.0)
            self._render_frame = render
            self._frame_open = True
            self._set_eyes_active(render)

            loc = self._head_loc
            rt.locate(rt.view_space, t, loc)
            bits = xr.SPACE_LOCATION_ORIENTATION_VALID_BIT
            self.head_tracked = bool(loc.location_flags & bits)
            if self.head_tracked:
                apply_pose(self.head, loc.pose, scale)

            if self.input is not None:
                self.input.update(t, scale)
                self.gaze_valid = self.input.gaze_valid
            if self.hands is not None:
                try:
                    self.hands.update(t, scale)
                except XrCallError as e:
                    log.warning("Hand tracking disabled: %s", e)
                    self.hands.destroy()
                    self.hands = None
        except XrCallError as e:
            log.warning("%s", e)
            self._frame_open = False
            self._teardown = "instance"
            return task.cont

        if self._mask_dirty:
            self._build_visibility_masks()
        self._sync_clear_color()
        self._update_stats(fs, render, wait_ms)
        return task.cont

    def _sync_clear_color(self):
        c = self.base.win.get_clear_color()
        if self.passthrough_active:
            c = (c[0], c[1], c[2], 0.0)
        c = tuple(c)
        if c != self._clear_color:
            self._clear_color = c
            for eye in self._eyes:
                if eye.buffer is not None:
                    eye.buffer.set_clear_color(c)

    def _build_visibility_masks(self):
        from .geometry import make_geom_node

        self._mask_dirty = False
        rt = self.rt
        if not (self.use_visibility_mask and rt is not None and rt.session is not None
                and rt.has("visibility_mask")):
            return
        for eye in self._eyes:
            if self._mask_data[eye.index] is None:
                try:
                    self._mask_data[eye.index] = rt.visibility_mask(eye.index)
                except Exception as e:
                    log.info("Visibility mask unavailable: %s", e)
                    self.use_visibility_mask = False
                    return
            verts, idx = self._mask_data[eye.index]
            if eye.mask is not None:
                eye.mask.remove_node()
                eye.mask = None
            if not len(idx):
                continue
            # Tangent-space (x, y) on the z=-1 plane -> just past the near plane.
            d = self.near * self.world_scale * 1.001
            v = verts.astype("float32")
            pos = [(x * d, d, y * d) for x, y in v]
            node = make_geom_node("vr-hidden-area-%d" % eye.index, pos, idx.reshape(-1, 3))
            mask = eye.cam.attach_new_node(node)
            mask.hide(EYE_MASKS[1 - eye.index])
            mask.set_bin(_MASK_BIN, 0)
            mask.set_depth_write(True)
            mask.set_depth_test(False)
            mask.set_two_sided(True)
            mask.set_color((0, 0, 0, 1), 10000)
            mask.set_light_off(10000)
            mask.set_texture_off(10000)
            mask.set_shader_off(10000)
            mask.set_fog_off(10000)
            mask.set_material_off(10000)
            node.set_final(True)
            eye.mask = mask

    def _update_stats(self, fs, rendered, wait_ms):
        st = self.stats
        st["frames"] += 1
        if not rendered:
            st["skipped"] += 1
        st["xr_wait_ms"] = st["xr_wait_ms"] * 0.9 + wait_ms * 0.1
        if fs is not None and fs.predicted_display_period:
            st["refresh_rate"] = 1e9 / fs.predicted_display_period
        st["fps"] = ClockObject.get_global_clock().get_average_frame_rate()
        if self._stats_text is not None:
            now = time.perf_counter()
            if now >= self._stats_next:
                self._stats_next = now + 0.25
                if fs is None:
                    text = "Desktop simulator (%s)  |  %.1f fps" % (self.status, st["fps"])
                else:
                    text = "%s  %.0f Hz  |  %.1f fps  |  xrWaitFrame %.2f ms  |  %dx%d  |  skipped %d" % (
                        self.system_name or "-", st["refresh_rate"], st["fps"], st["xr_wait_ms"],
                        st["eye_size"][0], st["eye_size"][1], st["skipped"])
                self._stats_text.setText(text)


_LEGACY_IGNORED = (
    "lensResolution", "FOV", "autoCamPositioning", "autoCamRotation",
    "autoControllerPositioning", "autoControllerRotation", "launchShowBase",
)


class BaseVrApp(ShowBase):
    """
    Drop-in replacement for ShowBase with VR attached as ``self.vr``.

    All VRManager keyword arguments are accepted.  The old constructor
    arguments (``wantVr``, ``wantDevMode``, ``lensResolution``...) still work:
    resolution, FOV and eye separation now come from the runtime, and tracking
    is always automatic.
    """

    def __init__(self, vr=True, debug_keys=False, **kwargs):
        if "wantVr" in kwargs:
            vr = kwargs.pop("wantVr")
        if kwargs.pop("wantDevMode", False):
            debug_keys = True
            kwargs.setdefault("show_stats", True)
        ignored = [k for k in _LEGACY_IGNORED if kwargs.pop(k, None) is not None]
        if ignored:
            log.info("Ignoring legacy options (now automatic): %s", ", ".join(ignored))

        # The headset paces the frame loop; desktop vsync would fight it.
        loadPrcFileData("panda3d-vr", "sync-video false\n")
        ShowBase.__init__(self)
        self.disableMouse()
        self.setBackgroundColor(0, 0, 0)
        self.vr = VRManager(self, enabled=bool(vr), **kwargs)
        if debug_keys:
            self.vr.enable_debug_keys()

    # ------------------------------------------------ legacy attribute names
    @property
    def player(self):
        return self.vr.rig

    @property
    def vrCam(self):
        return self.vr.head

    @property
    def hand_left(self):
        return self.vr.left.grip

    @property
    def hand_right(self):
        return self.vr.right.grip

    @property
    def cam_left(self):
        return self.vr.eye_cameras[0]

    @property
    def cam_right(self):
        return self.vr.eye_cameras[1]

    @property
    def HandState(self):
        return self.vr.controllers

    def haptic_feedback(self, hand, amplitude, duration, frequency):
        """Legacy signature: ``duration`` in nanoseconds."""
        self.vr.controllers[int(hand)].vibrate(amplitude, duration * 1e-9 if duration > 0 else -1, frequency)

    def reset_view_orientation(self):
        self.vr.recenter()

    resetView = reset_view_orientation

    def UpdateHeadsetTracking(self):
        """No longer needed: tracking updates automatically every frame."""
