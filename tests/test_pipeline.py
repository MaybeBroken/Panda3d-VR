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

    def create_session(self, hdc, hglrc, tracking):
        self.context = (hdc, hglrc)
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
    assert rt.context is not None and all(rt.context), "session must be created inside Panda's GL context"
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
    for i, eye in enumerate(vr._eyes):
        assert eye.mask is not None
        assert eye.mask.is_hidden(core.EYE_MASKS[1 - i])
        assert not eye.mask.is_hidden(core.EYE_MASKS[i])
    assert vr._eyes[0].mask.get_bin_name() == core._MASK_BIN


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
