"""
Low level OpenXR plumbing: instance, system, session, spaces, swapchains,
events and the frame loop.  No Panda3D in here.

Everything that runs per frame uses pyopenxr's raw ctypes entry points with
structures allocated once up front, so the steady state allocates nothing.
"""

import logging
from ctypes import POINTER, byref, c_float, c_uint32, c_void_p, cast, pointer

import numpy as np

import xr
from xr import raw_functions as raw

log = logging.getLogger("panda3d_vr")

ST = xr.StructureType
_EVENT_UNAVAILABLE = xr.Result.EVENT_UNAVAILABLE.value

# Optional features -> the extension that provides them.  Each is only enabled
# when the runtime advertises it.
FEATURE_EXTENSIONS = {
    "depth": "XR_KHR_composition_layer_depth",
    "visibility_mask": "XR_KHR_visibility_mask",
    "hand_tracking": "XR_EXT_hand_tracking",
    "hand_interaction": "XR_EXT_hand_interaction",
    "eye_gaze": "XR_EXT_eye_gaze_interaction",
    "refresh_rate": "XR_FB_display_refresh_rate",
    "performance": "XR_EXT_performance_settings",
    "local_floor": "XR_EXT_local_floor",
    "user_presence": "XR_EXT_user_presence",
    "passthrough": "XR_FB_passthrough",
    "debug": "XR_EXT_debug_utils",
}

_SPACE_TYPES = {
    "view": xr.ReferenceSpaceType.VIEW.value,
    "local": xr.ReferenceSpaceType.LOCAL.value,
    "stage": xr.ReferenceSpaceType.STAGE.value,
    "local_floor": xr.ReferenceSpaceType.LOCAL_FLOOR.value,
}
_SPACE_PREFERENCE = {
    "local_floor": ("local_floor", "stage", "local"),
    "stage": ("stage", "local_floor", "local"),
    "local": ("local",),
}

_PERF_DOMAINS = {"cpu": 1, "gpu": 2}
_PERF_LEVELS = {"power_savings": 0, "sustained_low": 25, "sustained_high": 50, "boost": 75}

_BLEND_MODES = {
    "opaque": xr.EnvironmentBlendMode.OPAQUE.value,
    "additive": xr.EnvironmentBlendMode.ADDITIVE.value,
    "alpha_blend": xr.EnvironmentBlendMode.ALPHA_BLEND.value,
}


class XrCallError(RuntimeError):
    def __init__(self, what, code):
        try:
            name = xr.Result(code).name
        except ValueError:
            name = str(code)
        super().__init__("%s failed: %s" % (what, name))
        self.code = code
        self.result_name = name


def check(result, what):
    code = int(getattr(result, "value", result))
    if code < 0:
        raise XrCallError(what, code)
    return code


def _enum_value(v):
    return int(getattr(v, "value", v))


class Swapchain:
    """One OpenXR swapchain plus its GL image names and reusable call structs."""

    __slots__ = (
        "handle", "width", "height", "format", "images",
        "_acquire", "_wait", "_release", "_index", "acquired",
    )

    def __init__(self, session, fmt, width, height, usage):
        info = xr.SwapchainCreateInfo(
            usage_flags=usage,
            format=fmt,
            sample_count=1,
            width=width,
            height=height,
            face_count=1,
            array_size=1,
            mip_count=1,
        )
        self.handle = xr.create_swapchain(session, info)
        self.width = width
        self.height = height
        self.format = fmt
        self.images = [
            int(img.image)
            for img in xr.enumerate_swapchain_images(self.handle, xr.SwapchainImageOpenGLKHR)
        ]
        self._acquire = xr.SwapchainImageAcquireInfo()
        self._wait = xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION)
        self._release = xr.SwapchainImageReleaseInfo()
        self._index = c_uint32(0)
        self.acquired = False

    def acquire(self):
        """Acquire + wait; returns the GL texture name to write into."""
        check(raw.xrAcquireSwapchainImage(self.handle, byref(self._acquire), byref(self._index)),
              "xrAcquireSwapchainImage")
        self.acquired = True
        check(raw.xrWaitSwapchainImage(self.handle, byref(self._wait)), "xrWaitSwapchainImage")
        return self.images[self._index.value]

    def release(self):
        if self.acquired:
            self.acquired = False
            check(raw.xrReleaseSwapchainImage(self.handle, byref(self._release)),
                  "xrReleaseSwapchainImage")

    def sub_image(self, target):
        """Fill an ``xr.SwapchainSubImage`` to cover this whole swapchain."""
        target.swapchain = self.handle
        target.image_rect.offset.x = 0
        target.image_rect.offset.y = 0
        target.image_rect.extent.width = self.width
        target.image_rect.extent.height = self.height
        target.image_array_index = 0

    def destroy(self):
        if self.handle is not None:
            try:
                xr.destroy_swapchain(self.handle)
            except Exception as e:  # pragma: no cover
                log.debug("destroy_swapchain: %s", e)
            self.handle = None


class XrRuntime:
    """
    Owns the OpenXR instance and session.  The Panda side (``core.VRManager``)
    drives it; this class only knows about OpenXR.

    ``on_event(name, **data)`` is called for every runtime event that matters
    to the app (session state changes, recentering, refresh-rate changes...).
    """

    def __init__(
        self,
        app_name="Panda3D VR",
        features=tuple(FEATURE_EXTENSIONS),
        extra_extensions=(),
        debug=False,
        blend_mode="opaque",
        on_event=None,
    ):
        self.app_name = app_name
        self.requested_features = set(features)
        if not debug:
            self.requested_features.discard("debug")
        self.extra_extensions = tuple(extra_extensions)
        self.debug = debug
        self.requested_blend_mode = blend_mode
        self.on_event = on_event or (lambda name, **data: None)

        self.view_config = xr.ViewConfigurationType.PRIMARY_STEREO
        self.instance = None
        self.system_id = None
        self.session = None
        self.space = None
        self.view_space = None
        self.space_type = None
        self.state = xr.SessionState.UNKNOWN
        self.running = False
        self.enabled_extensions = set()
        self.runtime_name = ""
        self.runtime_version = ""
        self.system_name = ""
        self.view_configs = []
        self.blend_mode = _BLEND_MODES["opaque"]
        self.max_layers = 16
        self.swapchain_formats = []
        self.reference_spaces = set()

        self._fns = {}
        self._messenger = None
        self._debug_cb = None

        # Reusable per-frame structures.
        self.frame_state = xr.FrameState()
        self._wait_info = xr.FrameWaitInfo()
        self._begin_info = xr.FrameBeginInfo()
        self._end_info = xr.FrameEndInfo()
        self._layer_ptrs = (POINTER(xr.CompositionLayerBaseHeader) * 16)()
        self._end_info._layers = cast(self._layer_ptrs, POINTER(POINTER(xr.CompositionLayerBaseHeader)))
        self._event = xr.EventDataBuffer()
        self._view_state = xr.ViewState()
        self._view_locate = xr.ViewLocateInfo(view_configuration_type=self.view_config)
        self._view_count = c_uint32(0)
        self.views = None

    # ------------------------------------------------------------------ setup

    def has(self, feature):
        ext = FEATURE_EXTENSIONS.get(feature, feature)
        return ext in self.enabled_extensions

    def create_instance(self):
        available = {
            e.extension_name.decode() for e in xr.enumerate_instance_extension_properties()
        }
        if xr.KHR_OPENGL_ENABLE_EXTENSION_NAME not in available:
            raise RuntimeError("The active OpenXR runtime does not support OpenGL")
        wanted = [xr.KHR_OPENGL_ENABLE_EXTENSION_NAME]
        for feature in sorted(self.requested_features):
            ext = FEATURE_EXTENSIONS.get(feature)
            if ext in available:
                wanted.append(ext)
        for ext in self.extra_extensions:
            if ext in available and ext not in wanted:
                wanted.append(ext)
            elif ext not in available:
                log.warning("Requested extension %s is not supported by the runtime", ext)

        app_info = xr.ApplicationInfo(
            application_name=self.app_name[:127],
            application_version=1,
            engine_name="Panda3D",
            engine_version=1,
            api_version=xr.XR_API_VERSION_1_0,
        )
        self.instance = xr.create_instance(
            xr.InstanceCreateInfo(application_info=app_info, enabled_extension_names=wanted)
        )
        self.enabled_extensions = set(wanted)
        props = xr.get_instance_properties(self.instance)
        self.runtime_name = props.runtime_name.decode(errors="replace")
        v = props.runtime_version
        self.runtime_version = str(v)
        log.info("OpenXR runtime: %s %s", self.runtime_name, self.runtime_version)
        log.debug("Enabled extensions: %s", ", ".join(wanted))
        if self.debug and self.has("debug"):
            self._create_debug_messenger()

    def try_get_system(self):
        """Returns False while no headset is available (unplugged, Link off...)."""
        try:
            self.system_id = xr.get_system(
                self.instance, xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY)
            )
        except xr.FormFactorUnavailableError:
            return False
        props = xr.get_system_properties(self.instance, self.system_id)
        self.system_name = props.system_name.decode(errors="replace")
        self.max_layers = min(16, int(props.graphics_properties.max_layer_count))
        self.position_tracking = bool(props.tracking_properties.position_tracking)
        self.view_configs = list(
            xr.enumerate_view_configuration_views(self.instance, self.system_id, self.view_config)
        )
        n = len(self.view_configs)
        self.views = (xr.View * n)(*[xr.View() for _ in range(n)])

        modes = [_enum_value(m) for m in xr.enumerate_environment_blend_modes(
            self.instance, self.system_id, self.view_config)]
        want = _BLEND_MODES.get(self.requested_blend_mode, _BLEND_MODES["opaque"])
        self.blend_mode = want if want in modes else modes[0]
        if want not in modes:
            log.warning("Blend mode %r unsupported, using %s", self.requested_blend_mode,
                        xr.EnvironmentBlendMode(self.blend_mode).name)

        # Calling this is mandatory before xrCreateSession with a GL binding.
        req = xr.GraphicsRequirementsOpenGLKHR()
        pfn = self.fn("xrGetOpenGLGraphicsRequirementsKHR", xr.PFN_xrGetOpenGLGraphicsRequirementsKHR)
        check(pfn(self.instance, self.system_id, byref(req)), "xrGetOpenGLGraphicsRequirementsKHR")
        self.gl_min_version = (req._min_api_version_supported >> 48) & 0xFFFF, \
            (req._min_api_version_supported >> 32) & 0xFFFF

        log.info("Headset: %s, %d views, recommended %dx%d per eye", self.system_name, n,
                 self.view_configs[0].recommended_image_rect_width,
                 self.view_configs[0].recommended_image_rect_height)
        return True

    def fn(self, name, pfn_type):
        """Resolve (and cache) an extension entry point."""
        f = self._fns.get(name)
        if f is None:
            f = cast(xr.get_instance_proc_addr(self.instance, name), pfn_type)
            self._fns[name] = f
        return f

    def _create_debug_messenger(self):
        levels = {0x1: logging.DEBUG, 0x10: logging.INFO, 0x100: logging.WARNING, 0x1000: logging.ERROR}

        def callback(severity, types, data, user_data):
            try:
                msg = data.contents.message
                log.log(levels.get(int(severity), logging.INFO), "[OpenXR] %s",
                        msg.decode(errors="replace") if msg else "")
            except Exception:
                pass
            return 0

        self._debug_cb = xr.PFN_xrDebugUtilsMessengerCallbackEXT(callback)
        info = xr.DebugUtilsMessengerCreateInfoEXT(
            message_severities=xr.DebugUtilsMessageSeverityFlagsEXT(0x1111),
            message_types=xr.DebugUtilsMessageTypeFlagsEXT(0xF),
            user_callback=self._debug_cb,
        )
        self._messenger = xr.DebugUtilsMessengerEXT()
        pfn = self.fn("xrCreateDebugUtilsMessengerEXT", xr.PFN_xrCreateDebugUtilsMessengerEXT)
        try:
            check(pfn(self.instance, byref(info), byref(self._messenger)),
                  "xrCreateDebugUtilsMessengerEXT")
        except XrCallError as e:
            log.warning("%s", e)
            self._messenger = None

    # ---------------------------------------------------------------- session

    def create_session(self, hdc, hglrc, tracking="local_floor"):
        binding = xr.GraphicsBindingOpenGLWin32KHR(h_dc=hdc, h_glrc=hglrc)
        info = xr.SessionCreateInfo(
            next=cast(pointer(binding), c_void_p), system_id=self.system_id
        )
        self.session = xr.create_session(self.instance, info)
        self.state = xr.SessionState.IDLE
        self.reference_spaces = {_enum_value(s) for s in xr.enumerate_reference_spaces(self.session)}
        self.swapchain_formats = [int(f) for f in xr.enumerate_swapchain_formats(self.session)]
        self.set_tracking_space(tracking)
        self.view_space = self.create_reference_space("view")
        self._view_locate.space = self.space

    def create_reference_space(self, kind, pose=None):
        info = xr.ReferenceSpaceCreateInfo(
            reference_space_type=xr.ReferenceSpaceType(_SPACE_TYPES[kind]),
            pose_in_reference_space=pose or xr.Posef(),
        )
        return xr.create_reference_space(self.session, info)

    def set_tracking_space(self, preferred):
        for kind in _SPACE_PREFERENCE.get(preferred, (preferred,)):
            if _SPACE_TYPES[kind] in self.reference_spaces:
                old = self.space
                self.space = self.create_reference_space(kind)
                self.space_type = kind
                self._view_locate.space = self.space
                if old is not None:
                    xr.destroy_space(old)
                if kind != preferred:
                    log.info("Tracking space %r unavailable, using %r", preferred, kind)
                return kind
        raise RuntimeError("No usable reference space")

    def destroy_session(self):
        for attr in ("view_space", "space"):
            s = getattr(self, attr)
            if s is not None:
                try:
                    xr.destroy_space(s)
                except Exception:
                    pass
                setattr(self, attr, None)
        if self.session is not None:
            try:
                xr.destroy_session(self.session)
            except Exception as e:
                log.debug("destroy_session: %s", e)
        self.session = None
        self.running = False
        self.state = xr.SessionState.UNKNOWN

    def destroy(self):
        self.destroy_session()
        if self._messenger is not None:
            try:
                pfn = self.fn("xrDestroyDebugUtilsMessengerEXT", xr.PFN_xrDestroyDebugUtilsMessengerEXT)
                pfn(self._messenger)
            except Exception:
                pass
            self._messenger = None
        if self.instance is not None:
            try:
                xr.destroy_instance(self.instance)
            except Exception:
                pass
        self.instance = None
        self.system_id = None
        self._fns.clear()

    def request_exit(self):
        if self.session is not None and self.running:
            xr.request_exit_session(self.session)

    # ----------------------------------------------------------------- events

    def poll_events(self):
        buf = self._event
        emit = self.on_event
        while self.instance is not None:
            buf.type = ST.EVENT_DATA_BUFFER.value
            buf.next = None
            if check(raw.xrPollEvent(self.instance, byref(buf)), "xrPollEvent") == _EVENT_UNAVAILABLE:
                return
            t = buf.type
            if t == ST.EVENT_DATA_SESSION_STATE_CHANGED.value:
                ev = cast(byref(buf), POINTER(xr.EventDataSessionStateChanged)).contents
                self._on_state(xr.SessionState(ev.state))
            elif t == ST.EVENT_DATA_INSTANCE_LOSS_PENDING.value:
                emit("instance-lost")
            elif t == ST.EVENT_DATA_INTERACTION_PROFILE_CHANGED.value:
                emit("interaction-profile-changed")
            elif t == ST.EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING.value:
                ev = cast(byref(buf), POINTER(xr.EventDataReferenceSpaceChangePending)).contents
                emit("reference-space-changed", space_type=ev.reference_space_type)
            elif t == ST.EVENT_DATA_VISIBILITY_MASK_CHANGED_KHR.value:
                ev = cast(byref(buf), POINTER(xr.EventDataVisibilityMaskChangedKHR)).contents
                emit("visibility-mask-changed", view_index=ev.view_index)
            elif t == ST.EVENT_DATA_DISPLAY_REFRESH_RATE_CHANGED_FB.value:
                ev = cast(byref(buf), POINTER(xr.EventDataDisplayRefreshRateChangedFB)).contents
                emit("refresh-rate-changed", old=ev.from_display_refresh_rate,
                     new=ev.to_display_refresh_rate)
            elif t == ST.EVENT_DATA_USER_PRESENCE_CHANGED_EXT.value:
                ev = cast(byref(buf), POINTER(xr.EventDataUserPresenceChangedEXT)).contents
                emit("user-presence", present=bool(ev.is_user_present))
            elif t == ST.EVENT_DATA_PERF_SETTINGS_EXT.value:
                ev = cast(byref(buf), POINTER(xr.EventDataPerfSettingsEXT)).contents
                emit("performance-notification", domain=ev.domain, sub_domain=ev.sub_domain,
                     level=ev.to_level)
            elif t == ST.EVENT_DATA_EVENTS_LOST.value:
                log.warning("OpenXR event queue overflowed; some events were lost")

    def _on_state(self, state):
        self.state = state
        log.debug("Session state -> %s", state.name)
        if state == xr.SessionState.READY:
            xr.begin_session(self.session, xr.SessionBeginInfo(self.view_config))
            self.running = True
        elif state == xr.SessionState.STOPPING:
            self.running = False
            xr.end_session(self.session)
        self.on_event("session-state", state=state)

    @property
    def focused(self):
        return self.state == xr.SessionState.FOCUSED

    @property
    def visible(self):
        return self.state in (xr.SessionState.VISIBLE, xr.SessionState.FOCUSED)

    # ------------------------------------------------------------------ frame

    def wait_and_begin_frame(self):
        fs = self.frame_state
        check(raw.xrWaitFrame(self.session, byref(self._wait_info), byref(fs)), "xrWaitFrame")
        check(raw.xrBeginFrame(self.session, byref(self._begin_info)), "xrBeginFrame")
        return fs

    def locate_views(self, time):
        """Returns (view_state_flags, views) for ``time``; views are reused."""
        self._view_locate.display_time = time
        n = len(self.views)
        check(raw.xrLocateViews(self.session, byref(self._view_locate), byref(self._view_state),
                                n, byref(self._view_count), self.views), "xrLocateViews")
        return self._view_state.view_state_flags, self.views

    def locate(self, space, time, location, base=None):
        check(raw.xrLocateSpace(space, base or self.space, time, byref(location)), "xrLocateSpace")
        return location

    def end_frame(self, time, layers):
        """``layers`` is a sequence of header pointers (see ``layer_pointer``)."""
        n = len(layers)
        for i in range(n):
            self._layer_ptrs[i] = layers[i]
        info = self._end_info
        info.display_time = time
        info.environment_blend_mode = self.blend_mode
        info.layer_count = n
        check(raw.xrEndFrame(self.session, byref(info)), "xrEndFrame")

    @staticmethod
    def layer_pointer(layer):
        return cast(pointer(layer), POINTER(xr.CompositionLayerBaseHeader))

    # ---------------------------------------------------------- swapchains

    def pick_color_format(self, preference):
        for f in preference:
            if f in self.swapchain_formats:
                return f
        return self.swapchain_formats[0]

    def create_swapchain(self, fmt, width, height, depth=False):
        F = xr.SwapchainUsageFlags
        usage = F.TRANSFER_DST_BIT | F.SAMPLED_BIT
        usage |= F.DEPTH_STENCIL_ATTACHMENT_BIT if depth else F.COLOR_ATTACHMENT_BIT
        return Swapchain(self.session, fmt, width, height, usage)

    # --------------------------------------------------------- extensions

    def visibility_mask(self, view_index, mask_type=xr.VisibilityMaskTypeKHR.HIDDEN_TRIANGLE_MESH):
        """Hidden-area mesh for one eye as (vertices[N,2] tangent-space, indices[M])."""
        pfn = self.fn("xrGetVisibilityMaskKHR", xr.PFN_xrGetVisibilityMaskKHR)
        mask = xr.VisibilityMaskKHR()
        args = (self.session, self.view_config.value, view_index, mask_type.value)
        check(pfn(*args, byref(mask)), "xrGetVisibilityMaskKHR")
        nv, ni = mask.vertex_count_output, mask.index_count_output
        if not nv or not ni:
            return np.zeros((0, 2), np.float32), np.zeros(0, np.uint32)
        idx_type = dict(xr.VisibilityMaskKHR._fields_)["indices"]._type_
        verts = (xr.Vector2f * nv)()
        idx = (idx_type * ni)()
        mask.vertex_capacity_input = nv
        mask.vertices = cast(verts, POINTER(xr.Vector2f))
        mask.index_capacity_input = ni
        mask.indices = cast(idx, POINTER(idx_type))
        check(pfn(*args, byref(mask)), "xrGetVisibilityMaskKHR")
        return (
            np.frombuffer(verts, np.float32).reshape(-1, 2).copy(),
            np.frombuffer(idx, np.uint32).copy(),
        )

    def refresh_rates(self):
        if not self.has("refresh_rate") or self.session is None:
            return []
        pfn = self.fn("xrEnumerateDisplayRefreshRatesFB", xr.PFN_xrEnumerateDisplayRefreshRatesFB)
        count = c_uint32(0)
        check(pfn(self.session, 0, byref(count), None), "xrEnumerateDisplayRefreshRatesFB")
        rates = (c_float * count.value)()
        check(pfn(self.session, count.value, byref(count), rates), "xrEnumerateDisplayRefreshRatesFB")
        return [round(r, 2) for r in rates]

    def refresh_rate(self):
        if not self.has("refresh_rate") or self.session is None:
            return None
        pfn = self.fn("xrGetDisplayRefreshRateFB", xr.PFN_xrGetDisplayRefreshRateFB)
        rate = c_float(0)
        check(pfn(self.session, byref(rate)), "xrGetDisplayRefreshRateFB")
        return round(rate.value, 2)

    def request_refresh_rate(self, hz):
        if not self.has("refresh_rate") or self.session is None:
            return False
        pfn = self.fn("xrRequestDisplayRefreshRateFB", xr.PFN_xrRequestDisplayRefreshRateFB)
        check(pfn(self.session, float(hz)), "xrRequestDisplayRefreshRateFB")
        return True

    def set_performance_level(self, domain, level):
        if not self.has("performance") or self.session is None:
            return False
        pfn = self.fn("xrPerfSettingsSetPerformanceLevelEXT", xr.PFN_xrPerfSettingsSetPerformanceLevelEXT)
        check(pfn(self.session, _PERF_DOMAINS[domain], _PERF_LEVELS[level]),
              "xrPerfSettingsSetPerformanceLevelEXT")
        return True

    def play_area(self):
        """(width, depth) of the guardian/chaperone rectangle in metres, or None."""
        if self.session is None:
            return None
        try:
            ext = xr.get_reference_space_bounds_rect(self.session, xr.ReferenceSpaceType.STAGE)
        except Exception:
            return None
        if ext.width <= 0 or ext.height <= 0:
            return None
        return ext.width, ext.height

    def path(self, s):
        return xr.string_to_path(self.instance, s)

    def path_str(self, p):
        return xr.path_to_string(self.instance, p)
