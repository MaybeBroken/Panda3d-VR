"""
Minimal OpenGL helpers that move Panda3D render textures into OpenXR swapchain
images entirely on the GPU.

Everything in here must be called while Panda3D's GL context is current, which
in practice means from inside a DisplayRegion draw callback.
"""

import ctypes
import sys

import OpenGL

# Per-call glGetError checks cost more than the GL calls themselves.  This only
# takes effect if nothing imported OpenGL.GL before us, which is the normal case.
OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False
OpenGL.ERROR_ON_COPY = False

if sys.platform.startswith("linux"):
    # Panda's context is EGL on Linux (see below), so GL entry points must be
    # resolved through EGL.  pyopenxr forces PyOpenGL onto GLX when imported,
    # so import it first and then switch back, before OpenGL.GL binds anything.
    import OpenGL.platform
    import xr  # noqa: F401
    from OpenGL.platform.egl import EGLPlatform

    if not isinstance(OpenGL.platform.PLATFORM, EGLPlatform):
        OpenGL.platform.PLATFORM = EGLPlatform()

from OpenGL import GL  # noqa: E402

GL_RGBA8 = 0x8058
GL_SRGB8_ALPHA8 = 0x8C43
GL_RGB10_A2 = 0x8059
GL_RGBA16F = 0x881A
GL_R11F_G11F_B10F = 0x8C3A
GL_DEPTH_COMPONENT16 = 0x81A5
GL_DEPTH_COMPONENT24 = 0x81A6
GL_DEPTH_COMPONENT32 = 0x81A7
GL_DEPTH_COMPONENT32F = 0x8CAC
GL_DEPTH24_STENCIL8 = 0x88F0
GL_DEPTH32F_STENCIL8 = 0x8CAD

# Swapchain colour formats in order of preference.  sRGB first: Panda writes
# display-referred (gamma encoded) values, and an sRGB swapchain tells the
# compositor exactly that, so colours are not double-encoded.
COLOR_FORMAT_PREFERENCE = (GL_SRGB8_ALPHA8, GL_RGBA8, GL_RGB10_A2, GL_RGBA16F)
DEPTH_FORMATS = {
    GL_DEPTH_COMPONENT16,
    GL_DEPTH_COMPONENT24,
    GL_DEPTH_COMPONENT32,
    GL_DEPTH_COMPONENT32F,
    GL_DEPTH24_STENCIL8,
    GL_DEPTH32F_STENCIL8,
}

# glCopyImageSubData requires "view compatible" formats (same texel size class).
_VIEW_CLASS = {
    GL_RGBA8: 32,
    GL_SRGB8_ALPHA8: 32,
    GL_RGB10_A2: 32,
    GL_R11F_G11F_B10F: 32,
    GL_RGBA16F: 64,
}


def copy_compatible(src_format, dst_format):
    if src_format == dst_format:
        return True
    a = _VIEW_CLASS.get(src_format)
    return a is not None and a == _VIEW_CLASS.get(dst_format)


if sys.platform == "win32":
    _wgl = ctypes.WinDLL("opengl32")
    _wgl.wglGetCurrentDC.restype = ctypes.c_void_p
    _wgl.wglGetCurrentContext.restype = ctypes.c_void_p
    _wgl.wglMakeCurrent.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    _wgl.wglMakeCurrent.restype = ctypes.c_int

    GRAPHICS_EXTENSIONS = ()

    def current_context():
        """(HDC, HGLRC) of the context current on this thread."""
        return _wgl.wglGetCurrentDC(), _wgl.wglGetCurrentContext()

    def ensure_current(ctx):
        """Some runtimes switch contexts inside xr calls; put Panda's back."""
        hdc, hglrc = ctx
        if _wgl.wglGetCurrentContext() != hglrc:
            _wgl.wglMakeCurrent(hdc, hglrc)

    def graphics_binding(ctx):
        import xr

        hdc, hglrc = ctx
        return xr.GraphicsBindingOpenGLWin32KHR(h_dc=hdc, h_glrc=hglrc)

elif sys.platform.startswith("linux"):
    # Linux goes through EGL (XR_MNDX_egl_enable) rather than GLX: Monado and
    # WiVRn only take GLX frames via GL_EXT_memory_object_fd, which some
    # drivers (Asahi on Apple Silicon) lack, while their EGL path can fall back
    # to dma-buf EGLImages.  Panda must therefore render with p3headlessgl.
    GRAPHICS_EXTENSIONS = ("XR_MNDX_egl_enable",)

    _EGL_CONFIG_ID = 0x3028
    _EGL_DRAW = 0x3059
    _EGL_READ = 0x305A

    _egl = ctypes.CDLL("libEGL.so.1")
    for _name in ("eglGetCurrentDisplay", "eglGetCurrentContext", "eglGetCurrentSurface"):
        getattr(_egl, _name).restype = ctypes.c_void_p
    _egl.eglGetCurrentSurface.argtypes = (ctypes.c_int32,)
    _egl.eglQueryContext.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
                                     ctypes.POINTER(ctypes.c_int32))
    _egl.eglChooseConfig.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                                     ctypes.POINTER(ctypes.c_void_p), ctypes.c_int32,
                                     ctypes.POINTER(ctypes.c_int32))
    _egl.eglMakeCurrent.argtypes = (ctypes.c_void_p,) * 4
    _egl.eglGetProcAddress.restype = ctypes.c_void_p
    _egl.eglGetProcAddress.argtypes = (ctypes.c_char_p,)

    def current_context():
        """(EGLDisplay, EGLConfig, EGLContext, draw, read) current on this thread."""
        ctx = _egl.eglGetCurrentContext()
        if not ctx:
            raise RuntimeError(
                "No EGL context is current.  On Linux Panda must render through EGL: "
                "put 'load-display p3headlessgl' and 'window-type offscreen' in your "
                "PRC config (BaseVrApp does this automatically)")
        dpy = _egl.eglGetCurrentDisplay()
        config_id = ctypes.c_int32(0)
        _egl.eglQueryContext(dpy, ctx, _EGL_CONFIG_ID, ctypes.byref(config_id))
        config = ctypes.c_void_p()
        if config_id.value:
            attribs = (ctypes.c_int32 * 3)(_EGL_CONFIG_ID, config_id.value, 0x3038)  # EGL_NONE
            count = ctypes.c_int32(0)
            _egl.eglChooseConfig(dpy, attribs, ctypes.byref(config), 1, ctypes.byref(count))
            if not count.value:
                config = ctypes.c_void_p()
        # A context made with EGL_KHR_no_config_context has no config; the
        # runtime only needs one for its own shared context, and accepts NULL.
        return (dpy, config.value, ctx,
                _egl.eglGetCurrentSurface(_EGL_DRAW), _egl.eglGetCurrentSurface(_EGL_READ))

    def ensure_current(ctx):
        """Some runtimes switch contexts inside xr calls; put Panda's back."""
        dpy, _config, context, draw, read = ctx
        if _egl.eglGetCurrentContext() != context:
            _egl.eglMakeCurrent(dpy, draw, read, context)

    def graphics_binding(ctx):
        import xr

        dpy, config, context = ctx[:3]
        binding = xr.GraphicsBindingEGLMNDX()
        binding.get_proc_address = xr.PFN_xrEglGetProcAddressMNDX(
            ctypes.cast(_egl.eglGetProcAddress, ctypes.c_void_p).value)
        # The handle fields are PyOpenGL's opaque EGL pointer types.
        fields = dict(xr.GraphicsBindingEGLMNDX._fields_)
        binding.display = ctypes.cast(dpy, fields["display"])
        binding.config = ctypes.cast(config, fields["config"])
        binding.context = ctypes.cast(context, fields["context"])
        return binding

else:  # pragma: no cover

    GRAPHICS_EXTENSIONS = ()

    def current_context():
        raise NotImplementedError("OpenXR is only implemented for Windows and Linux")

    def ensure_current(ctx):
        pass

    def graphics_binding(ctx):
        raise NotImplementedError("OpenXR is only implemented for Windows and Linux")


def gl_version():
    try:
        return GL.glGetString(GL.GL_VERSION).decode(errors="replace")
    except Exception:
        return "unknown"


_BINDINGS = {GL.GL_TEXTURE_2D: GL.GL_TEXTURE_BINDING_2D, GL.GL_TEXTURE_2D_ARRAY: GL.GL_TEXTURE_BINDING_2D_ARRAY}


def texture_internal_format(tex_id, target=GL.GL_TEXTURE_2D):
    """Query a texture's internal format without disturbing Panda's bindings.

    Returns 0 when the texture has no storage yet."""
    prev = int(GL.glGetIntegerv(_BINDINGS[target]))
    GL.glBindTexture(target, tex_id)
    width = int(GL.glGetTexLevelParameteriv(target, 0, GL.GL_TEXTURE_WIDTH))
    fmt = int(GL.glGetTexLevelParameteriv(target, 0, GL.GL_TEXTURE_INTERNAL_FORMAT))
    GL.glBindTexture(target, prev)
    return fmt if width else 0


def clear_errors():
    for _ in range(16):
        if GL.glGetError() == GL.GL_NO_ERROR:
            break


def last_error():
    return GL.glGetError()


class TextureCopier:
    """
    Copies one 2D texture into another of the same size.

    Uses glCopyImageSubData (a single call, touches no GL state) when the two
    formats are view compatible, and falls back to an FBO blit otherwise.
    Depth copies are only done between identical formats.
    """

    def __init__(self, src_format, dst_format, depth=False):
        self.depth = depth
        if copy_compatible(src_format, dst_format):
            self.mode = "copy"
        elif depth:
            raise ValueError(
                "depth formats differ (0x%04X vs 0x%04X)" % (src_format, dst_format)
            )
        else:
            self.mode = "blit"
        self._fbos = None

    def copy(self, src, dst, width, height):
        if self.mode == "copy":
            GL.glCopyImageSubData(
                src, GL.GL_TEXTURE_2D, 0, 0, 0, 0,
                dst, GL.GL_TEXTURE_2D, 0, 0, 0, 0,
                width, height, 1,
            )
        else:
            self._blit(src, dst, width, height)

    def _blit(self, src, dst, width, height):
        if self._fbos is None:
            self._fbos = [int(f) for f in GL.glGenFramebuffers(2)]
        read_fbo, draw_fbo = self._fbos
        prev_read = int(GL.glGetIntegerv(GL.GL_READ_FRAMEBUFFER_BINDING))
        prev_draw = int(GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING))
        scissor = GL.glIsEnabled(GL.GL_SCISSOR_TEST)
        srgb = GL.glIsEnabled(GL.GL_FRAMEBUFFER_SRGB)
        if scissor:
            GL.glDisable(GL.GL_SCISSOR_TEST)
        if srgb:
            GL.glDisable(GL.GL_FRAMEBUFFER_SRGB)

        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, read_fbo)
        GL.glFramebufferTexture2D(
            GL.GL_READ_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, src, 0
        )
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, draw_fbo)
        GL.glFramebufferTexture2D(
            GL.GL_DRAW_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, dst, 0
        )
        GL.glBlitFramebuffer(
            0, 0, width, height, 0, 0, width, height,
            GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST,
        )

        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, prev_read)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, prev_draw)
        if scissor:
            GL.glEnable(GL.GL_SCISSOR_TEST)
        if srgb:
            GL.glEnable(GL.GL_FRAMEBUFFER_SRGB)

    def destroy(self):
        if self._fbos:
            try:
                GL.glDeleteFramebuffers(2, self._fbos)
            except Exception:
                pass
            self._fbos = None
