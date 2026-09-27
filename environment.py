"""
The real world, as the Quest sees it.

EnvironmentDepth  (XR_META_environment_depth)
    The live depth map from the headset's depth sensors, refreshed every frame
    on the GPU into ``texture``: a Panda 2D texture array, layer 0 = left eye,
    layer 1 = right eye.  Each layer comes with the pose/FOV of the depth
    camera that produced it.  Optional automatic occlusion writes it into each
    eye's depth buffer before the scene draws, so real furniture, walls and
    hands hide virtual objects behind them -- with any shaders.

SceneModel  (XR_FB_scene + XR_META_spatial_entity_mesh)
    The room captured by Space Setup: the triangle mesh of the whole room plus
    labelled planes (floor, ceiling, walls, doors, windows) and volumes
    (tables, couches, beds...), as Panda geometry in tracking space.

Over Link both need the "Passthrough over Meta Quest Link" / "Spatial data"
developer options enabled in the Meta Quest Link app.
"""

import logging
import math
from time import perf_counter as _perf_counter
from ctypes import POINTER, addressof, byref, c_uint32, c_void_p, cast, create_string_buffer, pointer

import numpy as np
from panda3d.core import (
    Camera,
    CardMaker,
    ColorWriteAttrib,
    DepthTestAttrib,
    LMatrix4f,
    NodePath,
    OrthographicLens,
    Shader,
    Texture,
    TransparencyAttrib,
)
from direct.showbase.MessengerGlobal import messenger

import xr

from . import _gl
from .geometry import box, make_geom_node
from .runtime import check
from .xrmath import apply_pose

log = logging.getLogger("panda3d_vr")

_ENV_NOT_AVAILABLE = xr.Result.ENVIRONMENT_DEPTH_NOT_AVAILABLE_META.value

# Runtime depth formats -> (component type, Panda format).
_DEPTH_FORMATS = {
    _gl.GL_DEPTH_COMPONENT16: (Texture.T_unsigned_short, Texture.F_depth_component16),
    _gl.GL_DEPTH_COMPONENT24: (Texture.T_unsigned_int, Texture.F_depth_component24),
    _gl.GL_DEPTH_COMPONENT32F: (Texture.T_float, Texture.F_depth_component32),
}

# Panda local axes -> OpenXR view axes (row-vector convention), and back.
_PANDA_TO_XR = LMatrix4f(1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1)
_XR_TO_PANDA = LMatrix4f(1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1)

GLSL_LINEAR_DEPTH = """
// Environment depth texel -> metres along the depth camera's forward axis.
// env_depth_params comes from EnvironmentDepth.params (standard GL depth,
// far plane possibly infinite).
float env_linear_depth(float d, vec2 env_depth_params) {
    return env_depth_params.x / (d * 2.0 - 1.0 + env_depth_params.y);
}
"""

_KEEPALIVE_VERT = """#version 330
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
void main() { gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex; }
"""
_KEEPALIVE_FRAG = """#version 330
uniform sampler2DArray env_depth;
out vec4 color;
void main() { color = vec4(texture(env_depth, vec3(0.5, 0.5, 0.0)).r); }
"""

_OCCLUSION_VERT = """#version 330
in vec4 p3d_Vertex;
out vec2 ndc;
void main() {
    ndc = p3d_Vertex.xz;
    gl_Position = vec4(p3d_Vertex.xz, 0.0, 1.0);
}
"""
# For each eye pixel: take the view ray, look it up in the depth camera (the
# rotation between the two -- sensor latency -- is handled exactly; the few-cm
# translation is ignored), linearise, and write the real surface's depth.
_OCCLUSION_FRAG = """#version 330
uniform sampler2DArray env_depth;
uniform int env_layer;
uniform mat4 eye_clip_to_env;   // eye clip space -> depth-camera view space (XR axes)
uniform mat4 env_to_eye_clip;   // depth-camera view space -> eye clip space
uniform vec4 env_tan;           // tan(left), tan(right), tan(down), tan(up)
uniform vec2 env_params;
uniform float env_bias;         // metres pulled toward the viewer (reduces z-fighting)
uniform int env_flip;           // 1 if the runtime's image rows run top-down
in vec2 ndc;
""" + GLSL_LINEAR_DEPTH + """
void main() {
    vec4 far_point = eye_clip_to_env * vec4(ndc, 1.0, 1.0);
    vec3 dir = far_point.xyz / far_point.w;
    if (dir.z >= 0.0) discard;
    vec2 t = dir.xy / -dir.z;
    vec2 uv = (t - env_tan.xz) / (env_tan.yw - env_tan.xz);
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) discard;
    if (env_flip != 0) uv.y = 1.0 - uv.y;
    float d = texture(env_depth, vec3(uv, float(env_layer))).r;
    if (d >= 1.0 || d <= 0.0) discard;
    float z = max(env_linear_depth(d, env_params) - env_bias, 0.01);
    vec4 clip = env_to_eye_clip * vec4(t * z, -z, 1.0);
    gl_FragDepth = clamp(clip.z / clip.w * 0.5 + 0.5, 0.0, 1.0);
}
"""


def depth_params(near, far):
    """(invDepthFactor, depthOffset) so that metres = x / (ndc + y)."""
    if far < near or math.isinf(far):
        return -2.0 * near, -1.0
    return -2.0 * far * near / (near - far), (far + near) / (near - far)


class EnvironmentDepth:
    def __init__(self, vr):
        self.vr = vr
        self.rt = None
        self.texture = None
        self.width = self.height = 0
        self.valid = False
        self.near = 0.1
        self.far = float("inf")
        self.params = depth_params(self.near, self.far)
        self.fov = [(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)]
        self.views = [vr.tracking_space.attach_new_node("vr-env-depth-view-%d" % i) for i in range(2)]
        self.frame = 0
        self.hand_removal = False
        self.occlusion = False
        self.occlusion_bias = 0.02
        #: Diagnostics switches: skip acquiring / GPU-copying depth images.
        self.acquire_enabled = True
        self.copy_enabled = True
        self.acquire_ms = 0.0
        #: Set if the depth image turns out to be stored top row first.
        self.flip_y = False
        self.provider = None
        self.swapchain = None
        self._images = []
        self._format = None
        self._pending = None
        self._storage_ready = False
        self._occluders = []

    # ----------------------------------------------------------- lifecycle

    def create(self, rt, keepalive_root):
        """Inside the GL context, after the session exists."""
        self.rt = rt
        provider = xr.EnvironmentDepthProviderMETA()
        fn = rt.fn("xrCreateEnvironmentDepthProviderMETA", xr.PFN_xrCreateEnvironmentDepthProviderMETA)
        check(fn(rt.session, byref(xr.EnvironmentDepthProviderCreateInfoMETA()), byref(provider)),
              "xrCreateEnvironmentDepthProviderMETA")
        self.provider = provider
        swapchain = xr.EnvironmentDepthSwapchainMETA()
        fn = rt.fn("xrCreateEnvironmentDepthSwapchainMETA", xr.PFN_xrCreateEnvironmentDepthSwapchainMETA)
        check(fn(provider, byref(xr.EnvironmentDepthSwapchainCreateInfoMETA()), byref(swapchain)),
              "xrCreateEnvironmentDepthSwapchainMETA")
        self.swapchain = swapchain
        state = xr.EnvironmentDepthSwapchainStateMETA()
        fn = rt.fn("xrGetEnvironmentDepthSwapchainStateMETA", xr.PFN_xrGetEnvironmentDepthSwapchainStateMETA)
        check(fn(swapchain, byref(state)), "xrGetEnvironmentDepthSwapchainStateMETA")
        self.width, self.height = int(state.width), int(state.height)

        fn = rt.fn("xrEnumerateEnvironmentDepthSwapchainImagesMETA",
                   xr.PFN_xrEnumerateEnvironmentDepthSwapchainImagesMETA)
        count = c_uint32(0)
        check(fn(swapchain, 0, byref(count), None), "xrEnumerateEnvironmentDepthSwapchainImagesMETA")
        images = (xr.SwapchainImageOpenGLKHR * count.value)(
            *[xr.SwapchainImageOpenGLKHR() for _ in range(count.value)])
        check(fn(swapchain, count.value, byref(count),
                 cast(images, POINTER(xr.SwapchainImageBaseHeader))),
              "xrEnumerateEnvironmentDepthSwapchainImagesMETA")
        self._images = [int(i.image) for i in images]
        self._keepalive_root = keepalive_root
        # The runtime only allocates its images once the provider runs, so the
        # format -- and with it our own texture -- is settled on the first copy.
        self._format = None

        check(rt.fn("xrStartEnvironmentDepthProviderMETA", xr.PFN_xrStartEnvironmentDepthProviderMETA)(provider),
              "xrStartEnvironmentDepthProviderMETA")
        if self.hand_removal:
            self.set_hand_removal(True)
        log.info("Environment depth: %dx%d x2, %d images", self.width, self.height, len(self._images))

    def _make_texture(self, src):
        fmt = _gl.texture_internal_format(src, _gl.GL.GL_TEXTURE_2D_ARRAY)
        if not fmt:
            return False  # not allocated yet
        if fmt not in _DEPTH_FORMATS:
            raise RuntimeError("unexpected environment depth format 0x%04X" % fmt)
        self._format = fmt
        ctype, pfmt = _DEPTH_FORMATS[fmt]
        tex = Texture("vr-environment-depth")
        tex.setup_2d_texture_array(self.width, self.height, 2, ctype, pfmt)
        tex.set_minfilter(Texture.FT_nearest)
        tex.set_magfilter(Texture.FT_nearest)
        tex.set_wrap_u(Texture.WM_clamp)
        tex.set_wrap_v(Texture.WM_clamp)
        tex.make_ram_image()  # zeros; Panda uploads it, which allocates the GPU storage
        self.texture = tex
        # Drawn every frame by the prepare buffer so Panda keeps the texture resident.
        card = self._keepalive_root.attach_new_node(CardMaker("vr-env-depth-keepalive").generate())
        card.set_pos(-0.5, 1, -0.5)
        card.set_shader(Shader.make(Shader.SL_GLSL, _KEEPALIVE_VERT, _KEEPALIVE_FRAG))
        card.set_shader_input("env_depth", tex)
        self._keepalive = card
        log.info("Environment depth format 0x%04X", fmt)
        return True

    def destroy(self):
        rt = self.rt
        if rt is not None and self.provider is not None:
            for name, pfn, handle in (
                ("xrStopEnvironmentDepthProviderMETA", xr.PFN_xrStopEnvironmentDepthProviderMETA, self.provider),
                ("xrDestroyEnvironmentDepthSwapchainMETA", xr.PFN_xrDestroyEnvironmentDepthSwapchainMETA,
                 self.swapchain),
                ("xrDestroyEnvironmentDepthProviderMETA", xr.PFN_xrDestroyEnvironmentDepthProviderMETA,
                 self.provider),
            ):
                if handle is not None:
                    try:
                        rt.fn(name, pfn)(handle)
                    except Exception:
                        pass
        self.provider = self.swapchain = None
        self.valid = False
        self._storage_ready = False
        self._pending = None
        if getattr(self, "_keepalive", None) is not None:
            self._keepalive.remove_node()
            self._keepalive = None

    def set_hand_removal(self, enabled):
        """Remove the user's hands from the depth map (so they don't occlude)."""
        self.hand_removal = enabled
        if self.provider is None:
            return
        fn = self.rt.fn("xrSetEnvironmentDepthHandRemovalMETA", xr.PFN_xrSetEnvironmentDepthHandRemovalMETA)
        try:
            check(fn(self.provider, byref(xr.EnvironmentDepthHandRemovalSetInfoMETA(enabled=int(enabled)))),
                  "xrSetEnvironmentDepthHandRemovalMETA")
        except Exception as e:
            log.info("Hand removal unavailable: %s", e)

    # ------------------------------------------------------------- per frame

    def acquire(self, time, scale):
        """Frame task: fetch the newest depth image's metadata."""
        if self.provider is None or not self.acquire_enabled:
            return
        info = xr.EnvironmentDepthImageAcquireInfoMETA(space=self.rt.space, display_time=time)
        image = xr.EnvironmentDepthImageMETA()
        for v in image.views:
            v.type = xr.StructureType.ENVIRONMENT_DEPTH_IMAGE_VIEW_META.value
        fn = self.rt.fn("xrAcquireEnvironmentDepthImageMETA", xr.PFN_xrAcquireEnvironmentDepthImageMETA)
        t0 = _perf_counter()
        code = check(fn(self.provider, byref(info), byref(image)), "xrAcquireEnvironmentDepthImageMETA")
        self.acquire_ms = (_perf_counter() - t0) * 1000.0
        if code == _ENV_NOT_AVAILABLE:
            return
        self.near = float(image.near_z)
        self.far = float(image.far_z)
        self.params = depth_params(self.near, self.far)
        for i in range(2):
            v = image.views[i]
            apply_pose(self.views[i], v.pose, scale)
            f = v.fov
            self.fov[i] = (math.tan(f.angle_left), math.tan(f.angle_right),
                           math.tan(f.angle_down), math.tan(f.angle_up))
        self._pending = int(image.swapchain_index)
        if self._storage_ready:
            self.valid = True
            self.frame += 1

    def copy(self, native_id):
        """GL callback (before the eyes render): runtime image -> Panda texture."""
        if self._pending is None or not self.copy_enabled:
            return
        src = self._images[self._pending]
        if self.texture is None:
            self._make_texture(src)
            return  # Panda allocates it on the next draw
        dst = native_id(self.texture)
        if not self._storage_ready:
            try:
                ready = _gl.texture_internal_format(dst, _gl.GL.GL_TEXTURE_2D_ARRAY) == self._format
            except Exception:
                ready = False
            if not ready:
                return  # uploaded by the keep-alive draw next frame
            self._storage_ready = True
        _gl.GL.glCopyImageSubData(
            src, _gl.GL.GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0,
            dst, _gl.GL.GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0,
            self.width, self.height, 2)
        self._pending = None

    # ------------------------------------------------------------ occlusion

    def set_occlusion(self, enabled, bias=None):
        """Let real-world depth hide virtual objects (best with passthrough)."""
        self.occlusion = enabled
        if bias is not None:
            self.occlusion_bias = bias
        for occ in self._occluders:
            occ["region"].set_active(enabled and self.valid)

    def attach_occluders(self, eyes):
        """Add a depth pre-pass display region to each eye buffer."""
        shader = Shader.make(Shader.SL_GLSL, _OCCLUSION_VERT, _OCCLUSION_FRAG)
        for eye in eyes:
            scene = NodePath("vr-env-occlusion-%d" % eye.index)
            lens = OrthographicLens()
            lens.set_film_size(2, 2)
            lens.set_near_far(-10, 10)
            cam = scene.attach_new_node(Camera("vr-env-occlusion-cam-%d" % eye.index, lens))
            cm = CardMaker("occluder")
            cm.set_frame_fullscreen_quad()
            card = scene.attach_new_node(cm.generate())
            card.set_shader(shader)
            card.set_attrib(ColorWriteAttrib.make(ColorWriteAttrib.C_off))
            card.set_attrib(DepthTestAttrib.make(DepthTestAttrib.M_less))
            card.set_depth_write(True)
            card.set_two_sided(True)
            card.set_shader_input("env_layer", eye.index)
            region = eye.buffer.make_display_region()
            region.set_sort(-5)  # after the hidden-area mesh, before the scene
            region.set_camera(cam)
            region.set_active(False)
            self._occluders.append({"eye": eye, "card": card, "region": region})

    def update_occluders(self):
        """Frame task, after the eye cameras moved: refresh the reprojection."""
        on = self.occlusion and self.valid
        for occ in self._occluders:
            occ["region"].set_active(on)
            if not on:
                continue
            eye = occ["eye"]
            i = eye.index
            proj = LMatrix4f(eye.lens.get_projection_mat())
            inv_proj = LMatrix4f(proj)
            inv_proj.invert_in_place()
            eye_to_env = eye.cam.get_mat(self.views[i])
            env_to_eye = self.views[i].get_mat(eye.cam)
            card = occ["card"]
            card.set_shader_input("env_depth", self.texture)
            card.set_shader_input("eye_clip_to_env", inv_proj * eye_to_env * _PANDA_TO_XR)
            card.set_shader_input("env_to_eye_clip", _XR_TO_PANDA * env_to_eye * proj)
            l, r, d, u = self.fov[i]
            card.set_shader_input("env_tan", (l, r, d, u))
            card.set_shader_input("env_params", self.params)
            card.set_shader_input("env_bias", self.occlusion_bias * self.vr.world_scale)
            card.set_shader_input("env_flip", int(self.flip_y))

    def detach_occluders(self):
        for occ in self._occluders:
            occ["eye"].buffer.remove_display_region(occ["region"])
        self._occluders = []

    # ---------------------------------------------------------------- queries

    def shader_inputs(self, np):
        """Bind env_depth / env_depth_params / env_view_{0,1} / env_tan_{0,1} on ``np``.

        ``env_view_i`` maps *render* (world) space to depth-camera view space
        (XR axes, -Z forward) for a custom shader doing its own occlusion.
        Call every frame.
        """
        if self.texture is None:
            return
        render = self.vr.render
        np.set_shader_input("env_depth", self.texture)
        np.set_shader_input("env_depth_params", self.params)
        for i in range(2):
            np.set_shader_input("env_view_%d" % i, render.get_mat(self.views[i]) * _PANDA_TO_XR)
            np.set_shader_input("env_tan_%d" % i, self.fov[i])

    def read_depth(self):
        """Blocking GPU readback: (2, H, W) float32 metres (inf = no data).

        Row 0 is the *bottom* of the image (GL convention; see ``flip_y``).
        """
        if not self.valid:
            return None
        base = self.vr.base
        base.graphicsEngine.extract_texture_data(self.texture, base.win.get_gsg())
        ctype = self.texture.get_component_type()
        dtype = {Texture.T_unsigned_short: np.uint16, Texture.T_unsigned_int: np.uint32,
                 Texture.T_float: np.float32}[ctype]
        raw = np.frombuffer(memoryview(self.texture.get_ram_image()), dtype=dtype)
        d = raw.reshape(2, self.height, self.width).astype(np.float64)
        if dtype == np.uint16:
            d /= 65535.0
        elif dtype == np.uint32:
            d /= 4294967295.0
        x, y = self.params
        with np.errstate(divide="ignore", invalid="ignore"):
            metres = x / (d * 2.0 - 1.0 + y)
        metres[(d >= 1.0) | (d <= 0.0)] = np.inf
        return metres.astype(np.float32)

    def world_point(self, eye, u, v, metres):
        """World-space point for depth-image coords (u, v in 0..1, v up) at ``metres``."""
        if self.flip_y:
            v = 1.0 - v
        l, r, d, t = self.fov[eye]
        x = l + (r - l) * u
        y = d + (t - d) * v
        local = _XR_TO_PANDA.xform_point((x * metres, y * metres, -metres))
        return self.vr.render.get_relative_point(self.views[eye], local * self.vr.world_scale)


# =============================================================== scene model

SEMANTIC_LABELS = (
    "TABLE,COUCH,FLOOR,CEILING,WALL_FACE,WINDOW_FRAME,DOOR_FRAME,STORAGE,BED,SCREEN,"
    "LAMP,PLANT,WALL_ART,GLOBAL_MESH,INVISIBLE_WALL_FACE,OTHER"
)
_LABEL_COLORS = {
    "FLOOR": (0.3, 0.8, 0.3, 0.35), "CEILING": (0.6, 0.6, 0.9, 0.35),
    "WALL_FACE": (0.9, 0.9, 0.9, 0.25), "DOOR_FRAME": (0.9, 0.6, 0.2, 0.5),
    "WINDOW_FRAME": (0.3, 0.7, 1.0, 0.5), "TABLE": (0.9, 0.4, 0.2, 0.45),
    "COUCH": (0.7, 0.3, 0.7, 0.45), "BED": (0.8, 0.3, 0.4, 0.45),
}
_COMPONENT = xr.SpaceComponentTypeFB


class SceneAnchor:
    """One Space Setup entity: ``node`` is placed at the anchor's pose."""

    def __init__(self, space, uuid, parent):
        self.space = space
        self.uuid = uuid
        self.labels = []
        self.node = parent.attach_new_node("anchor-" + uuid[:8])
        self.mesh = None       # NodePath with the triangle mesh (GLOBAL_MESH)
        self.plane = None      # (x, y, width, height) in the anchor's plane, metres
        self.volume = None     # (x, y, z, width, height, depth), metres
        self.located = False

    @property
    def label(self):
        return self.labels[0] if self.labels else "UNKNOWN"

    def __repr__(self):
        return "<SceneAnchor %s %s>" % (self.label, self.uuid[:8])


class SceneModel:
    """
    The Space Setup room.  Call ``load()`` (or pass ``scene=True`` to
    VRManager); ``vr-scene-loaded`` fires when it's in.  Everything hangs off
    ``root`` (in tracking space) and is hidden until ``show()``.
    """

    def __init__(self, vr):
        self.vr = vr
        self.rt = None
        self.root = vr.tracking_space.attach_new_node("vr-scene")
        self.root.hide()
        self.anchors = {}
        self.loaded = False
        self._requests = {}
        self._relocate_at = 0.0

    @property
    def mesh(self):
        """NodePath of the room's global triangle mesh, or None."""
        for a in self.anchors.values():
            if a.mesh is not None:
                return a.mesh
        return None

    def by_label(self, label):
        return [a for a in self.anchors.values() if label in a.labels]

    def attach(self, rt):
        self.rt = rt

    def load(self):
        """Ask the runtime for the stored room (async)."""
        if self.rt is None or self.rt.session is None:
            return False
        self.loaded = False
        for kind in (_COMPONENT.TRIANGLE_MESH_M, _COMPONENT.BOUNDED_2D, _COMPONENT.BOUNDED_3D):
            location = xr.SpaceStorageLocationFilterInfoFB(location=xr.SpaceStorageLocationFB.LOCAL)
            comp = xr.SpaceComponentFilterInfoFB(
                next=cast(pointer(location), c_void_p), component_type=kind)
            info = xr.SpaceQueryInfoFB(
                query_action=xr.SpaceQueryActionFB.LOAD, max_result_count=256, timeout=0,
                filter=cast(pointer(comp), POINTER(xr.SpaceFilterInfoBaseHeaderFB)))
            request = xr.AsyncRequestIdFB()
            fn = self.rt.fn("xrQuerySpacesFB", xr.PFN_xrQuerySpacesFB)
            try:
                check(fn(self.rt.session, cast(byref(info), POINTER(xr.SpaceQueryInfoBaseHeaderFB)),
                         byref(request)), "xrQuerySpacesFB")
            except Exception as e:
                log.warning("Scene query failed: %s", e)
                return False
            self._requests[int(request.value)] = (kind, (info, comp, location))
        return True

    def request_capture(self):
        """Launch Space Setup so the user can (re)scan the room."""
        if self.rt is None or self.rt.session is None:
            return False
        fn = self.rt.fn("xrRequestSceneCaptureFB", xr.PFN_xrRequestSceneCaptureFB)
        request = xr.AsyncRequestIdFB()
        check(fn(self.rt.session, byref(xr.SceneCaptureRequestInfoFB()), byref(request)),
              "xrRequestSceneCaptureFB")
        return True

    def show(self):
        self.root.show()

    def hide(self):
        self.root.hide()

    # ------------------------------------------------------------ events

    def on_results(self, request_id):
        entry = self._requests.get(request_id)
        if entry is None:
            return
        kind = entry[0]
        rt = self.rt
        fn = rt.fn("xrRetrieveSpaceQueryResultsFB", xr.PFN_xrRetrieveSpaceQueryResultsFB)
        results = xr.SpaceQueryResultsFB()
        check(fn(rt.session, request_id, byref(results)), "xrRetrieveSpaceQueryResultsFB")
        n = results.result_count_output
        log.debug("Scene query %d (%s): %d results", request_id, kind.name, n)
        if not n:
            return
        arr = (xr.SpaceQueryResultFB * n)()
        results.result_capacity_input = n
        results.results = cast(arr, POINTER(xr.SpaceQueryResultFB))
        check(fn(rt.session, request_id, byref(results)), "xrRetrieveSpaceQueryResultsFB")
        for res in arr:
            uuid = bytes(res.uuid.data).hex()
            anchor = self.anchors.get(uuid)
            if anchor is None:
                anchor = self.anchors[uuid] = SceneAnchor(res.space, uuid, self.root)
                self._ensure_locatable(anchor)
                anchor.labels = self._labels(anchor)
            try:
                if kind == _COMPONENT.TRIANGLE_MESH_M:
                    self._load_mesh(anchor)
                elif kind == _COMPONENT.BOUNDED_2D:
                    self._load_plane(anchor)
                else:
                    self._load_volume(anchor)
            except Exception as e:
                log.debug("Scene anchor %s: %s", anchor, e)

    def on_complete(self, request_id):
        self._requests.pop(request_id, None)
        if not self._requests and not self.loaded:
            self.loaded = True
            self._relocate_at = 0.0
            log.info("Scene loaded: %d anchors (%s)", len(self.anchors),
                     ", ".join(sorted({a.label for a in self.anchors.values()})) or "empty")
            messenger.send("vr-scene-loaded", [self])

    # ----------------------------------------------------------- per frame

    def update(self, time, scale, now):
        if not self.anchors or now < self._relocate_at:
            return
        self._relocate_at = now + 2.0  # anchors are static; re-locate occasionally
        valid = xr.SPACE_LOCATION_POSITION_VALID_BIT | xr.SPACE_LOCATION_ORIENTATION_VALID_BIT
        loc = xr.SpaceLocation()
        for anchor in self.anchors.values():
            try:
                self.rt.locate(anchor.space, time, loc)
            except Exception:
                continue
            if (loc.location_flags & valid) == valid:
                apply_pose(anchor.node, loc.pose, scale)
                anchor.node.set_scale(scale)
                anchor.located = True

    def destroy(self):
        for anchor in self.anchors.values():
            try:
                xr.destroy_space(anchor.space)
            except Exception:
                pass
        self.anchors = {}
        self.root.node().remove_all_children()
        self.loaded = False
        self._requests = {}

    # ----------------------------------------------------------- helpers

    def _ensure_locatable(self, anchor):
        rt = self.rt
        status = xr.SpaceComponentStatusFB()
        get = rt.fn("xrGetSpaceComponentStatusFB", xr.PFN_xrGetSpaceComponentStatusFB)
        try:
            check(get(anchor.space, _COMPONENT.LOCATABLE.value, byref(status)), "xrGetSpaceComponentStatusFB")
        except Exception:
            return
        if not status.enabled and not status.change_pending:
            setter = rt.fn("xrSetSpaceComponentStatusFB", xr.PFN_xrSetSpaceComponentStatusFB)
            request = xr.AsyncRequestIdFB()
            info = xr.SpaceComponentStatusSetInfoFB(component_type=_COMPONENT.LOCATABLE, enabled=1, timeout=0)
            try:
                setter(anchor.space, byref(info), byref(request))
            except Exception:
                pass

    def _labels(self, anchor):
        rt = self.rt
        fn = rt.fn("xrGetSpaceSemanticLabelsFB", xr.PFN_xrGetSpaceSemanticLabelsFB)
        # Multiple labels | desk->table migration | invisible wall faces.
        support = xr.SemanticLabelsSupportInfoFB(flags=1 | 2 | 4, recognized_labels=SEMANTIC_LABELS)
        labels = xr.SemanticLabelsFB(next=cast(pointer(support), c_void_p))
        try:
            check(fn(rt.session, anchor.space, byref(labels)), "xrGetSpaceSemanticLabelsFB")
            n = labels.buffer_count_output
            if not n:
                return []
            buf = create_string_buffer(n + 1)
            labels.buffer_capacity_input = n
            # A c_char_p field copies on assignment; write the raw pointer instead.
            slot = cast(addressof(labels) + xr.SemanticLabelsFB.buffer.offset, POINTER(c_void_p))
            slot[0] = addressof(buf)
            check(fn(rt.session, anchor.space, byref(labels)), "xrGetSpaceSemanticLabelsFB")
        except Exception:
            return []
        return [s for s in buf.value.decode(errors="replace").split(",") if s]

    def _load_mesh(self, anchor):
        rt = self.rt
        fn = rt.fn("xrGetSpaceTriangleMeshMETA", xr.PFN_xrGetSpaceTriangleMeshMETA)
        info = xr.SpaceTriangleMeshGetInfoMETA()
        mesh = xr.SpaceTriangleMeshMETA()
        check(fn(anchor.space, byref(info), byref(mesh)), "xrGetSpaceTriangleMeshMETA")
        nv, ni = mesh.vertex_count_output, mesh.index_count_output
        if not nv or not ni:
            return
        idx_type = dict(xr.SpaceTriangleMeshMETA._fields_)["indices"]._type_
        verts = (xr.Vector3f * nv)()
        idx = (idx_type * ni)()
        mesh.vertex_capacity_input = nv
        mesh.vertices = cast(verts, POINTER(xr.Vector3f))
        mesh.index_capacity_input = ni
        mesh.indices = cast(idx, POINTER(idx_type))
        check(fn(anchor.space, byref(info), byref(mesh)), "xrGetSpaceTriangleMeshMETA")
        v = np.frombuffer(verts, np.float32).reshape(-1, 3)
        panda = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)  # XR -> Panda axes
        tris = np.frombuffer(idx, np.uint32).reshape(-1, 3)
        node = anchor.node.attach_new_node(make_geom_node("scene-mesh", panda, tris))
        node.set_render_mode_wireframe()
        node.set_color(0.4, 0.9, 1.0, 1)
        node.set_light_off(1)
        node.set_python_tag("vr_scene_anchor", anchor)
        anchor.mesh = node
        if "GLOBAL_MESH" not in anchor.labels:
            anchor.labels.append("GLOBAL_MESH")

    def _load_plane(self, anchor):
        fn = self.rt.fn("xrGetSpaceBoundingBox2DFB", xr.PFN_xrGetSpaceBoundingBox2DFB)
        rect = xr.Rect2Df()
        check(fn(self.rt.session, anchor.space, byref(rect)), "xrGetSpaceBoundingBox2DFB")
        x, y, w, h = rect.offset.x, rect.offset.y, rect.extent.width, rect.extent.height
        anchor.plane = (x, y, w, h)
        cm = CardMaker("scene-plane")
        cm.set_frame(x, x + w, y, y + h)  # XR plane (x, y) -> Panda (x, z), facing -Y
        card = anchor.node.attach_new_node(cm.generate())
        card.set_two_sided(True)
        card.set_color(_LABEL_COLORS.get(anchor.label, (1, 1, 1, 0.3)))
        card.set_transparency(TransparencyAttrib.M_alpha)
        card.set_light_off(1)
        card.set_python_tag("vr_scene_anchor", anchor)

    def _load_volume(self, anchor):
        fn = self.rt.fn("xrGetSpaceBoundingBox3DFB", xr.PFN_xrGetSpaceBoundingBox3DFB)
        rect = xr.Rect3DfFB()
        check(fn(self.rt.session, anchor.space, byref(rect)), "xrGetSpaceBoundingBox3DFB")
        o, e = rect.offset, rect.extent
        anchor.volume = (o.x, o.y, o.z, e.width, e.height, e.depth)
        # XR (x, y, z) box -> Panda (x, -z, y)
        center = (o.x + e.width / 2, -(o.z + e.depth / 2), o.y + e.height / 2)
        node = anchor.node.attach_new_node(box((e.width / 2, e.depth / 2, e.height / 2), center,
                                               _LABEL_COLORS.get(anchor.label, (1, 1, 1, 0.3)),
                                               "scene-volume"))
        node.set_transparency(TransparencyAttrib.M_alpha)
        node.set_light_off(1)
        node.set_python_tag("vr_scene_anchor", anchor)
