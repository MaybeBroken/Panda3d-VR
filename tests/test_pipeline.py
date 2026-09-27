"""
End-to-end test of the render -> swapchain pipeline without a headset.

A fake runtime stands in for OpenXR, but its swapchain images are real GL
textures, so these tests verify what actually reaches the compositor: frame
ordering, the GPU copy/blit paths and depth submission.
"""

from types import SimpleNamespace

import numpy as np
import pytest

xr = pytest.importorskip("xr")

import panda3d_vr.core as core  # noqa: E402
from panda3d_vr import _gl  # noqa: E402
from OpenGL import GL  # noqa: E402
from panda3d.core import CardMaker  # noqa: E402

W, H = 64, 48


class FakeSwapchain:
    def __init__(self, fmt, w, h, depth):
        self.format, self.width, self.height, self.depth = fmt, w, h, depth
        prev = int(GL.glGetIntegerv(GL.GL_TEXTURE_BINDING_2D))
        self.tex = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex)
        GL.glTexStorage2D(GL.GL_TEXTURE_2D, 1, fmt, w, h)
        GL.glBindTexture(GL.GL_TEXTURE_2D, prev)
        self.images = [self.tex]
        self.pixels = None
        self.releases = 0

    def acquire(self):
        return self.tex

    def release(self):
        prev = int(GL.glGetIntegerv(GL.GL_TEXTURE_BINDING_2D))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex)
        if self.depth:
            data = GL.glGetTexImage(GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)
            self.pixels = np.frombuffer(data, np.float32).reshape(self.height, self.width)
        else:
            data = GL.glGetTexImage(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, GL.GL_FLOAT)
            self.pixels = np.frombuffer(data, np.float32).reshape(self.height, self.width, 4)
        GL.glBindTexture(GL.GL_TEXTURE_2D, prev)
        self.releases += 1

    def sub_image(self, target):
        target.image_rect.extent.width = self.width
        target.image_rect.extent.height = self.height

    def destroy(self):
        pass


class FakeRuntime:
    def __init__(self, formats, features=()):
        self.system_id = 1
        self.system_name = "Fake HMD"
        self.runtime_name = "fake"
        self.session = None
        self.running = False
        self.focused = False
        self.state = xr.SessionState.IDLE
        self.space = self.view_space = None
        self.space_type = "local_floor"
        self.max_layers = 16
        self.swapchain_formats = formats
        self.features = set(features)
        vc = SimpleNamespace(recommended_image_rect_width=W, recommended_image_rect_height=H,
                             max_image_rect_width=4096, max_image_rect_height=4096)
        self.view_configs = [vc, vc]
        self.frame_state = xr.FrameState()
        self.frame_state.should_render = 1
        self.frame_state.predicted_display_time = 1
        self.frame_state.predicted_display_period = 11_111_111
        self.views = (xr.View * 2)(xr.View(), xr.View())
        for i, v in enumerate(self.views):
            v.pose.orientation.w = 1
            v.pose.position.x = (-0.032, 0.032)[i]
            v.pose.position.y = 1.6
            v.fov = xr.Fovf(-0.9, 0.8, 0.85, -0.95) if i == 0 else xr.Fovf(-0.8, 0.9, 0.85, -0.95)
        self.swapchains = []
        self.ended = []
        self.context = None

    def has(self, feature):
        return feature in self.features

    def create_session(self, binding, tracking):
        self.context = binding
        self.session = object()
        self.running = True

    def poll_events(self):
        pass

    def pick_color_format(self, preference):
        return next((f for f in preference if f in self.swapchain_formats), self.swapchain_formats[0])

    def create_swapchain(self, fmt, w, h, depth=False):
        sc = FakeSwapchain(fmt, w, h, depth)
        self.swapchains.append(sc)
        return sc

    def wait_and_begin_frame(self):
        self.frame_state.predicted_display_time += 11_111_111
        return self.frame_state

    def locate_views(self, time):
        return xr.VIEW_STATE_ORIENTATION_VALID_BIT | xr.VIEW_STATE_POSITION_VALID_BIT, self.views

    def locate(self, space, time, loc, base=None):
        loc.location_flags = 0xF
        loc.pose = self.views[0].pose
        return loc

    layer_pointer = staticmethod(core.XrRuntime.layer_pointer)

    def end_frame(self, time, layers):
        self.ended.append(len(layers))

    def refresh_rate(self):
        return 90.0

    def destroy(self):
        pass

    def destroy_session(self):
        pass


class FakeInput:
    gaze_valid = False

    def __init__(self, *a, **k):
        pass

    def create(self):
        pass

    def attach(self):
        pass

    def update(self, t, scale):
        pass

    def destroy(self):
        pass

    def refresh_profiles(self):
        pass


@pytest.fixture
def make_vr(base, monkeypatch):
    monkeypatch.setattr(core, "XrInput", FakeInput)
    made = []

    def factory(formats, features=(), **kw):
        base.setBackgroundColor(1, 0, 0, 1)
        vr = core.VRManager(base, fallback=None, show_controllers=False, hand_tracking=False, **kw)
        vr._try_connect = lambda: None
        vr.rt = FakeRuntime(formats, features)
        vr._on_system_ready()
        made.append(vr)
        return vr

    yield factory
    for vr in made:
        vr.destroy()


def _run(base, frames=4):
    for _ in range(frames):
        base.taskMgr.step()


def test_frames_reach_swapchain_via_copy(base, make_vr):
    vr = make_vr([_gl.GL_SRGB8_ALPHA8, _gl.GL_RGBA8], msaa=0, submit_depth=False)
    _run(base)
    rt = vr.rt
    assert rt.context is not None, "session must be created inside Panda's GL context"
    assert rt.ended and rt.ended[-1] == 1, "one projection layer per frame"
    assert [sc.format for sc in rt.swapchains] == [_gl.GL_SRGB8_ALPHA8] * 2
    assert all(e.copier.mode == "copy" for e in vr._eyes)
    for sc in rt.swapchains:
        px = sc.pixels
        assert px.shape == (H, W, 4)
        assert np.allclose(px[H // 2, W // 2, :3], (1, 0, 0), atol=0.02)


def test_eye_cameras_follow_views(base, make_vr):
    vr = make_vr([_gl.GL_RGBA8], msaa=0, submit_depth=False)
    _run(base)
    left, right = vr.eye_cameras
    assert abs(left.get_x() + 0.032) < 1e-6 and abs(right.get_x() - 0.032) < 1e-6
    assert abs(left.get_z() - 1.6) < 1e-6
    fw, fh, ox, oy = core.fov_to_film(vr.rt.views[0].fov)
    assert abs(vr._eyes[0].lens.get_film_offset()[0] - ox) < 1e-6


def test_blit_fallback_for_incompatible_format(base, make_vr):
    vr = make_vr([_gl.GL_RGBA16F], msaa=0, submit_depth=False)
    _run(base)
    assert all(e.copier.mode == "blit" for e in vr._eyes)
    px = vr.rt.swapchains[0].pixels
    assert np.allclose(px[H // 2, W // 2, :3], (1, 0, 0), atol=0.02)


def test_msaa_resolves_before_submit(base, make_vr):
    vr = make_vr([_gl.GL_SRGB8_ALPHA8], msaa=4, submit_depth=False)
    _run(base)
    px = vr.rt.swapchains[0].pixels
    assert np.allclose(px[H // 2, W // 2, :3], (1, 0, 0), atol=0.02)


@pytest.mark.parametrize("msaa", [0, 4])
def test_depth_submission(base, make_vr, msaa):
    formats = [_gl.GL_SRGB8_ALPHA8, _gl.GL_DEPTH_COMPONENT32F, _gl.GL_DEPTH_COMPONENT24,
               _gl.GL_DEPTH24_STENCIL8]
    vr = make_vr(formats, features={"depth"}, msaa=msaa, submit_depth=True)
    cm = CardMaker("wall")
    cm.set_frame(-50, 50, -50, 50)
    wall = base.render.attach_new_node(cm.generate())
    wall.set_pos(0, 3, 1.6)  # 3 m in front of the eyes, covering the view
    try:
        _run(base)
    finally:
        wall.remove_node()
    depth = [sc for sc in vr.rt.swapchains if sc.depth]
    assert len(depth) == 2, "one depth swapchain per eye"
    assert vr._proj_views[0].next, "depth info chained to the projection view"
    # Standard GL depth for a point 3 m away with near 0.05 / far 1000:
    near, far, z = 0.05, 1000.0, 3.0
    expected = (far / (far - near)) * (1 - near / z)
    assert abs(depth[0].pixels[H // 2, W // 2] - expected) < 1e-3, "resolved scene depth reaches the swapchain"


def test_skip_render_when_runtime_says_so(base, make_vr):
    vr = make_vr([_gl.GL_RGBA8], msaa=0, submit_depth=False)
    _run(base, 3)
    vr.rt.frame_state.should_render = 0
    _run(base, 2)
    assert vr.rt.ended[-1] == 0, "no layers submitted"
    assert not vr._eyes[0].buffer.is_active(), "eye buffers idle"
    vr.rt.frame_state.should_render = 1
    _run(base, 2)
    assert vr.rt.ended[-1] == 1 and vr._eyes[0].buffer.is_active()


def test_visibility_mask_geometry(base, make_vr):
    vr = make_vr([_gl.GL_RGBA8], features={"visibility_mask"}, msaa=0, submit_depth=False)
    verts = np.array([[-1, -1], [1, -1], [-1, -0.8]], np.float32)
    vr.rt.visibility_mask = lambda i: (verts, np.array([0, 1, 2], np.uint32))
    _run(base)
    for eye in vr._eyes:
        assert eye.mask is not None
        assert eye.mask.get_top() == eye.mask_root, "mask is outside the shared scene graph"


def test_quad_layer_submitted(base, make_vr):
    vr = make_vr([_gl.GL_SRGB8_ALPHA8], msaa=0, submit_depth=False)
    quad = vr.create_quad_layer((32, 16), (1.0, 0.5))
    quad.buffer.set_clear_color((0, 1, 0, 1))
    quad.node.set_pos(0, 2, 1.5)
    _run(base, 5)
    assert vr.rt.ended[-1] == 2
    assert abs(quad.layer.pose.position.z + 2.0) < 1e-5 and abs(quad.layer.pose.position.y - 1.5) < 1e-5
    assert np.allclose(quad.swapchain.pixels[8, 16, :3], (0, 1, 0), atol=0.02)


def test_quad_layer_created_after_session(base, make_vr):
    vr = make_vr([_gl.GL_SRGB8_ALPHA8], msaa=0, submit_depth=False)
    created = []
    vr.accept("vr-session-created", lambda v: created.append(v.create_quad_layer((32, 16))))
    _run(base, 5)
    assert created and created[0].swapchain is not None
    assert vr.rt.ended[-1] == 2


def test_hidden_area_mesh_only_in_its_own_eye(base, make_vr):
    """Regression: each eye's hidden-area mesh leaked into the other eye's view.

    The mesh sits just past the near plane of its own eye, so seen from the
    other eye (an IPD away) it covered the inner top/bottom corners -- the
    artifact reported next to the nose."""
    vr = make_vr([_gl.GL_RGBA8], features={"visibility_mask"}, msaa=0, submit_depth=False)
    # Left eye: a mask covering its whole frustum.  Right eye: no mask.
    full = np.array([[-3, -3], [3, -3], [3, 3], [-3, 3]], np.float32)
    quad = np.array([0, 1, 2, 0, 2, 3], np.uint32)
    vr.rt.visibility_mask = lambda i: (full, quad) if i == 0 else (np.zeros((0, 2), np.float32), np.zeros(0, np.uint32))
    # Put the other eye's mask where it would be visible: a wide IPD.
    vr.rt.views[0].pose.position.x = -0.03
    vr.rt.views[1].pose.position.x = 0.03
    _run(base, 5)
    left, right = vr.rt.swapchains[0].pixels, vr.rt.swapchains[1].pixels
    assert np.allclose(left[..., :3], 0, atol=0.02), "left eye fully masked"
    leaked = ~np.isclose(right[..., :3], (1, 0, 0), atol=0.02).all(axis=-1)
    assert not leaked.any(), "right eye sees the left mask in %d pixels" % leaked.sum()


# ---------------------------------------------------------- environment depth

ENV_W, ENV_H, ENV_NEAR, WALL = 32, 32, 0.1, 1.0


def _fake_env(vr):
    """Stand-in for XR_META_environment_depth: a real GL D16 texture array
    holding a flat wall WALL metres in front of each eye."""
    env = vr.environment_depth

    def create(rt, keepalive_root):
        env.rt = rt
        env.provider = object()
        env.width, env.height = ENV_W, ENV_H
        env._keepalive_root = keepalive_root
        env._format = None
        d = 1.0 - ENV_NEAR / WALL  # infinite-far GL depth of the wall
        data = np.full((2, ENV_H, ENV_W), int(d * 65535), np.uint16)
        prev = int(GL.glGetIntegerv(GL.GL_TEXTURE_BINDING_2D_ARRAY))
        tex = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)  # complete
        GL.glTexImage3D(GL.GL_TEXTURE_2D_ARRAY, 0, _gl.GL_DEPTH_COMPONENT16, ENV_W, ENV_H, 2, 0,
                        GL.GL_DEPTH_COMPONENT, GL.GL_UNSIGNED_SHORT, data)
        GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, prev)
        env._images = [tex]

    def acquire(time, scale):
        env.near, env.far = ENV_NEAR, float("inf")
        env.params = core_env.depth_params(env.near, env.far)
        for i in range(2):
            core.apply_pose(env.views[i], vr.rt.views[i].pose, scale)
            f = vr.rt.views[i].fov
            import math
            env.fov[i] = tuple(math.tan(a) for a in (f.angle_left, f.angle_right, f.angle_down, f.angle_up))
        env._pending = 0
        if env._storage_ready:
            env.valid = True
            env.frame += 1

    def destroy():
        env.provider = None
        env.valid = False

    class FakePassthrough:
        handle = ptr = None
        feed_running = False

        def start_feed(self):
            self.feed_running = True

        def destroy(self):
            pass

    env.create = create
    env.acquire = acquire
    env.destroy = destroy
    vr.rt.focused = True
    vr.passthrough = FakePassthrough()
    return env


SETTLE = 50  # frames: MR features wait for a focused, stable session


import panda3d_vr.environment as core_env  # noqa: E402


def test_environment_depth_copy_and_readback(base, make_vr):
    vr = make_vr([_gl.GL_RGBA8], features={"environment_depth"}, msaa=0, submit_depth=False,
                 environment_depth=True)
    _fake_env(vr)
    _run(base, SETTLE + 6)
    env = vr.environment_depth
    assert env.valid and env.texture is not None
    metres = env.read_depth()
    assert metres.shape == (2, ENV_H, ENV_W)
    assert np.allclose(metres, WALL, atol=0.01), "linearised depth matches the wall"
    p = env.world_point(0, 0.5, 0.5, WALL)
    eye = vr.eye_cameras[0].get_pos(base.render)
    assert abs((p - eye).length() - WALL) < 0.05


@pytest.mark.parametrize("distance, occluded", [(3.0, True), (0.5, False)])
def test_environment_occlusion(base, make_vr, distance, occluded):
    vr = make_vr([_gl.GL_RGBA8], features={"environment_depth"}, msaa=0, submit_depth=False,
                 environment_depth="occlusion")
    _fake_env(vr)
    cm = CardMaker("virtual")
    cm.set_frame(-5, 5, -5, 5)
    card = base.render.attach_new_node(cm.generate())
    card.set_color(0, 1, 0, 1)
    card.set_pos(0, distance, 1.6)  # a green virtual wall at `distance`
    try:
        _run(base, SETTLE + 8)
    finally:
        card.remove_node()
    px = vr.rt.swapchains[0].pixels[H // 2, W // 2, :3]
    if occluded:
        assert np.allclose(px, (1, 0, 0), atol=0.02), "real wall at 1 m hides the virtual one at 3 m"
    else:
        assert np.allclose(px, (0, 1, 0), atol=0.02), "virtual object in front of the real wall stays"


# ------------------------------------------------------------ external sources

def _external_eye(base, vr, i, color=(0, 1, 0, 1), float_color=False, depth=True):
    """A stand-in external renderer: its own buffer, drawing through the VR eye camera."""
    from panda3d.core import FrameBufferProperties, GraphicsOutput, GraphicsPipe, Texture, WindowProperties

    fb = FrameBufferProperties()
    if float_color:
        fb.set_float_color(True)
        fb.set_rgba_bits(16, 16, 16, 16)
    else:
        fb.set_rgba_bits(8, 8, 8, 8)
    fb.set_depth_bits(32)
    fb.set_float_depth(True)
    w, h = vr.eye_size
    buf = base.graphicsEngine.make_output(base.pipe, "external-%d" % i, -60, fb, WindowProperties.size(w, h),
                                          GraphicsPipe.BF_refuse_window, base.win.get_gsg(), base.win)
    tex, dtex = Texture("ext-color-%d" % i), Texture("ext-depth-%d" % i)
    buf.add_render_texture(tex, GraphicsOutput.RTM_bind_or_copy, GraphicsOutput.RTP_color)
    buf.add_render_texture(dtex, GraphicsOutput.RTM_bind_or_copy, GraphicsOutput.RTP_depth)
    buf.set_clear_color_active(True)
    buf.set_clear_color(color)
    buf.make_display_region().set_camera(vr.eye_cameras[i])
    vr.set_eye_source(i, tex, dtex if depth else None)
    return buf


def test_external_eye_sources(base, make_vr):
    formats = [_gl.GL_SRGB8_ALPHA8, _gl.GL_DEPTH_COMPONENT32F]
    vr = make_vr(formats, features={"depth"}, msaa=0, submit_depth=True)
    assert vr.eyes_ready and vr.eye_size == (W, H)
    bufs = [_external_eye(base, vr, 0, (0, 1, 0, 1)), _external_eye(base, vr, 1, (0, 0, 1, 1), float_color=True)]
    try:
        _run(base, 6)
        color = [sc for sc in vr.rt.swapchains if not sc.depth]
        assert np.allclose(color[0].pixels[H // 2, W // 2, :3], (0, 1, 0), atol=0.02)
        assert np.allclose(color[1].pixels[H // 2, W // 2, :3], (0, 0, 1), atol=0.02)
        assert vr._eyes[0].copier.mode == "copy" and vr._eyes[1].copier.mode == "blit"
        assert not vr._eyes[0].buffer.is_active(), "built-in eye rendering is idle"
        assert len([sc for sc in vr.rt.swapchains if sc.depth]) == 2, "external depth submitted"

        vr.clear_eye_sources()
        _run(base, 4)
        assert np.allclose(color[0].pixels[H // 2, W // 2, :3], (1, 0, 0), atol=0.02), "back to built-in"
    finally:
        for b in bufs:
            base.graphicsEngine.remove_window(b)


# ------------------------------------------------------- mcshader integration

def test_mcshader_stereo(base, make_vr):
    """A Minecraft shaderpack rendered per eye and submitted to the headset.

    Needs mcshader and a pack: set MCSHADER_PACK to a pack .zip/directory."""
    import os
    pack = os.environ.get("MCSHADER_PACK")
    mcshader = pytest.importorskip("mcshader")
    if not pack or not os.path.exists(pack):
        pytest.skip("set MCSHADER_PACK to run")
    from direct.showbase.MessengerGlobal import messenger  # noqa: F401

    vr = make_vr([_gl.GL_SRGB8_ALPHA8, _gl.GL_DEPTH_COMPONENT32F], features={"depth"},
                 msaa=0, submit_depth=True)
    app = mcshader.init(base, pack=pack, profile="LOW", sky=True, vr=vr)
    try:
        assert app.vr.active, "bridge switched the pipeline to the eyes"
        assert [v.name for v in app.pipe.views] == ["vr-eye0", "vr-eye1"]
        ground = app.load("models/environment", type="terrain", scale=0.25, pos=(-8, 42, 0))
        _run(base, 12)
        color = [sc for sc in vr.rt.swapchains if not sc.depth]
        depth = [sc for sc in vr.rt.swapchains if sc.depth]
        left, right = color[0].pixels[..., :3], color[1].pixels[..., :3]
        for img in (left, right):
            assert not np.allclose(img, (1, 0, 0), atol=0.05), "shaded image, not the clear colour"
            assert img.std() > 0.01, "a real scene, not a flat fill"
        assert np.abs(left - right).mean() > 1e-3, "each eye rendered from its own camera"
        assert len(depth) == 2 and (depth[0].pixels < 1.0).any(), "scene depth submitted"
        assert not vr._eyes[0].buffer.is_active()
        # Screen-only camera effects are off in the headset...
        opts = app.pipe.options
        for name, value in app.vr.option_overrides.items():
            if name in opts.options:
                assert str(opts.get(name)).lower() == str(value).lower(), name
        saved = dict(app.vr._saved_options)
        app.vr.detach()
        # ...and the user's own values come back with the window.
        for name, value in saved.items():
            assert str(opts.get(name)).lower() == str(value).lower(), name
    finally:
        app.vr.detach()
        base.taskMgr.remove("mcshader-pipeline-uniforms")
        app.pipe._teardown()
        if app.sky is not None:
            app.sky.remove_node()
        for np_ in base.render.find_all_matches("**/+ModelRoot"):
            np_.remove_node()


def test_environment_depth_waits_for_a_settled_session(base, make_vr):
    vr = make_vr([_gl.GL_RGBA8], features={"environment_depth"}, msaa=0, submit_depth=False,
                 environment_depth=True)
    env = _fake_env(vr)
    _run(base, 10)
    assert env.provider is None, "not started while the session is settling"
    _run(base, SETTLE)
    assert env.provider is not None and vr.passthrough.feed_running, "started with the camera feed"


def test_environment_depth_needs_passthrough(base, make_vr):
    vr = make_vr([_gl.GL_RGBA8], features={"environment_depth"}, msaa=0, submit_depth=False,
                 environment_depth=True)
    env = _fake_env(vr)
    vr.passthrough = None
    disabled = []
    base.accept("vr-feature-disabled", lambda name, why: disabled.append(name))
    _run(base, SETTLE + 5)
    base.ignore("vr-feature-disabled")
    assert env.provider is None and disabled == ["environment_depth"]


def test_stall_watchdog_disables_environment_depth(base, make_vr):
    import time
    vr = make_vr([_gl.GL_RGBA8], features={"environment_depth"}, msaa=0, submit_depth=False,
                 environment_depth=True)
    env = _fake_env(vr)
    vr.STALL_SECONDS = 0.05  # scaled down for the test
    real_acquire = env.acquire

    def stalling_acquire(t, scale):
        real_acquire(t, scale)
        time.sleep(0.08)  # the runtime choking on depth

    env.acquire = stalling_acquire
    _run(base, SETTLE + 8)
    assert env.provider is None, "watchdog switched depth off"
    assert "environment_depth" in vr._mr_failed and not vr.want_environment_depth
    before = len(vr.rt.ended)
    _run(base, 3)
    assert len(vr.rt.ended) == before + 3, "the app keeps rendering"


def test_depth_without_images_is_switched_off(base, make_vr):
    """Over Link without passthrough the runtime starts depth but never
    delivers an image, while costing frame time: stop it."""
    vr = make_vr([_gl.GL_RGBA8], features={"environment_depth"}, msaa=0, submit_depth=False,
                 environment_depth=True)
    env = _fake_env(vr)
    vr.DEPTH_DATA_TIMEOUT = 0.05
    env.acquire = lambda t, scale: None  # provider running, no images ever
    reasons = []
    base.accept("vr-feature-disabled", lambda name, why: reasons.append(why))
    import time
    for _ in range(SETTLE + 20):
        base.taskMgr.step()
        time.sleep(0.005)
    base.ignore("vr-feature-disabled")
    assert env.provider is None and reasons and "no depth images" in reasons[0]


def test_sustained_slowdown_switches_feature_off(base, make_vr):
    import time
    vr = make_vr([_gl.GL_RGBA8], features={"environment_depth"}, msaa=0, submit_depth=False,
                 environment_depth=True)
    env = _fake_env(vr)
    real_acquire = env.acquire
    started = []

    def slow_acquire(t, scale):
        real_acquire(t, scale)
        started.append(1)
        time.sleep(0.03)  # every frame now much slower, but never a single "stall"

    env.acquire = slow_acquire
    for _ in range(SETTLE):
        base.taskMgr.step()
        time.sleep(0.004)  # a steady baseline before depth starts
    t0 = time.perf_counter()
    while env.provider is not None or not started:
        base.taskMgr.step()
        assert time.perf_counter() - t0 < 10, "slowdown never detected"
    assert "environment_depth" in vr._mr_failed
