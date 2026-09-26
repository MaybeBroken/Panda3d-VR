"""
Extra OpenXR composition layers.

QuadLayer   - a flat panel composited by the runtime at native display
              resolution (crisp text/UI, independent of eye-buffer resolution).
Passthrough - Meta's camera passthrough (XR_FB_passthrough) behind the scene.
"""

import logging
from ctypes import byref

from panda3d.core import (
    Camera,
    ColorBlendAttrib,
    NodePath,
    OrthographicLens,
    Texture,
)

import xr

from . import _gl
from .runtime import check
from .xrmath import panda_to_pose

log = logging.getLogger("panda3d_vr")

_EYES = {"both": 0, "left": 1, "right": 2}


class QuadLayer:
    """
    A flat panel with its own 2D scene graph, submitted as an OpenXR quad layer.

    ``root`` works like ``aspect2d``: x spans [-aspect, aspect], z spans [-1, 1].
    ``node`` places the panel in the world; it faces -Y like a CardMaker card
    and ``size`` is its width and height in metres.

    Quad layers are composited over the 3D scene without depth testing, which
    is what you want for HUDs, menus and tooltips.
    """

    def __init__(self, vr, resolution=(1024, 512), size=(1.0, 0.5), parent=None,
                 name="quad-layer", eye="both", clear_color=(0, 0, 0, 0)):
        self.vr = vr
        self.name = name
        self.width, self.height = resolution
        self.size = size
        self.eye = eye
        self.visible = True
        self.node = (parent or vr.tracking_space).attach_new_node(name)

        self.root = NodePath(name + "-2d")
        self.root.set_depth_test(False)
        self.root.set_depth_write(False)
        self.root.set_two_sided(True)
        self.root.set_light_off(1)
        # Premultiplied alpha output so the compositor blends correctly.
        self.root.set_attrib(ColorBlendAttrib.make(
            ColorBlendAttrib.M_add, ColorBlendAttrib.O_incoming_alpha,
            ColorBlendAttrib.O_one_minus_incoming_alpha,
            ColorBlendAttrib.M_add, ColorBlendAttrib.O_one,
            ColorBlendAttrib.O_one_minus_incoming_alpha))

        self.texture = Texture(name)
        self.buffer = vr._make_buffer(name, self.width, self.height, sort=-40, msaa=0,
                                      color_tex=self.texture)
        self.buffer.set_clear_color_active(True)
        self.buffer.set_clear_color(clear_color)
        aspect = self.width / float(self.height)
        lens = OrthographicLens()
        lens.set_film_size(2 * aspect, 2)
        lens.set_near_far(-1000, 1000)
        cam = Camera(name + "-cam", lens)
        cam.set_scene(self.root)
        self.camera = self.root.attach_new_node(cam)
        dr = self.buffer.make_display_region()
        dr.set_camera(self.camera)
        self.aspect = aspect

        self.swapchain = None
        self._copier = None
        self._pose = xr.Posef()
        self.layer = None
        self._ptr = None

    # Called by VRManager inside the GL context.
    def _create(self, rt, native_id):
        src = native_id(self.texture)
        src_fmt = _gl.texture_internal_format(src)
        fmt = rt.pick_color_format(_gl.COLOR_FORMAT_PREFERENCE)
        self.swapchain = rt.create_swapchain(fmt, self.width, self.height)
        self._copier = _gl.TextureCopier(src_fmt, fmt)
        self.layer = xr.CompositionLayerQuad(
            layer_flags=xr.CompositionLayerFlags.BLEND_TEXTURE_SOURCE_ALPHA_BIT,
            space=rt.space,
            eye_visibility=xr.EyeVisibility(_EYES[self.eye]),
        )
        self.swapchain.sub_image(self.layer.sub_image)
        self._ptr = rt.layer_pointer(self.layer)

    def _submit(self, rt, native_id, scale):
        sc = self.swapchain
        dst = sc.acquire()
        self._copier.copy(native_id(self.texture), dst, sc.width, sc.height)
        sc.release()
        ts = self.node.get_transform(self.vr.tracking_space)
        s = ts.get_scale()
        layer = self.layer
        layer.space = rt.space
        layer.pose = panda_to_pose(ts.get_pos(), ts.get_norm_quat(), scale, self._pose)
        layer.size.width = self.size[0] * s[0]
        layer.size.height = self.size[1] * s[2]
        return self._ptr

    def _destroy_xr(self):
        if self.swapchain is not None:
            self.swapchain.destroy()
            self.swapchain = None
        self._copier = None
        self.layer = None

    def show(self):
        self.visible = True
        self.buffer.set_active(True)

    def hide(self):
        self.visible = False
        self.buffer.set_active(False)

    def destroy(self):
        self._destroy_xr()
        self.vr._remove_quad_layer(self)
        self.vr.base.graphicsEngine.remove_window(self.buffer)
        self.node.remove_node()
        self.root.remove_node()


class Passthrough:
    """Meta passthrough.  Over Link it needs "Passthrough over Link" enabled."""

    def __init__(self, rt, enabled=True):
        self.rt = rt
        self.enabled = enabled
        self.handle = None
        self.layer_handle = None
        self.layer = None
        self.ptr = None

    def create(self):
        rt = self.rt
        running = 1 if self.enabled else 0  # XR_PASSTHROUGH_IS_RUNNING_AT_CREATION_BIT_FB
        self.handle = xr.PassthroughFB()
        create = rt.fn("xrCreatePassthroughFB", xr.PFN_xrCreatePassthroughFB)
        check(create(rt.session, byref(xr.PassthroughCreateInfoFB(flags=running)), byref(self.handle)),
              "xrCreatePassthroughFB")
        self.layer_handle = xr.PassthroughLayerFB()
        create_layer = rt.fn("xrCreatePassthroughLayerFB", xr.PFN_xrCreatePassthroughLayerFB)
        info = xr.PassthroughLayerCreateInfoFB(
            passthrough=self.handle, flags=running,
            purpose=xr.PassthroughLayerPurposeFB.RECONSTRUCTION)
        check(create_layer(rt.session, byref(info), byref(self.layer_handle)),
              "xrCreatePassthroughLayerFB")
        self.layer = xr.CompositionLayerPassthroughFB(
            flags=xr.CompositionLayerFlags.BLEND_TEXTURE_SOURCE_ALPHA_BIT,
            layer_handle=self.layer_handle)
        self.ptr = rt.layer_pointer(self.layer)

    def set_enabled(self, enabled):
        self.enabled = enabled
        if self.handle is None:
            return
        rt = self.rt
        if enabled:
            check(rt.fn("xrPassthroughStartFB", xr.PFN_xrPassthroughStartFB)(self.handle),
                  "xrPassthroughStartFB")
            check(rt.fn("xrPassthroughLayerResumeFB", xr.PFN_xrPassthroughLayerResumeFB)(
                self.layer_handle), "xrPassthroughLayerResumeFB")
        else:
            check(rt.fn("xrPassthroughLayerPauseFB", xr.PFN_xrPassthroughLayerPauseFB)(
                self.layer_handle), "xrPassthroughLayerPauseFB")
            check(rt.fn("xrPassthroughPauseFB", xr.PFN_xrPassthroughPauseFB)(self.handle),
                  "xrPassthroughPauseFB")

    def set_opacity(self, opacity, edge_color=(0, 0, 0, 0)):
        if self.layer_handle is None:
            return
        style = xr.PassthroughStyleFB(texture_opacity_factor=float(opacity),
                                      edge_color=xr.Color4f(*edge_color))
        fn = self.rt.fn("xrPassthroughLayerSetStyleFB", xr.PFN_xrPassthroughLayerSetStyleFB)
        check(fn(self.layer_handle, byref(style)), "xrPassthroughLayerSetStyleFB")

    def destroy(self):
        rt = self.rt
        if self.layer_handle is not None:
            try:
                rt.fn("xrDestroyPassthroughLayerFB", xr.PFN_xrDestroyPassthroughLayerFB)(self.layer_handle)
            except Exception:
                pass
        if self.handle is not None:
            try:
                rt.fn("xrDestroyPassthroughFB", xr.PFN_xrDestroyPassthroughFB)(self.handle)
            except Exception:
                pass
        self.handle = self.layer_handle = self.layer = self.ptr = None
