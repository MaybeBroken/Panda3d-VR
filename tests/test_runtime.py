"""Checks against the real OpenXR runtime that work without a headset."""

import logging

import pytest

xr = pytest.importorskip("xr")

from panda3d_vr.controller import Controller  # noqa: E402
from panda3d_vr.input import HAND_INTERACTION_PROFILE, PROFILES, XrInput  # noqa: E402
from panda3d_vr.runtime import XrRuntime  # noqa: E402


@pytest.fixture
def rt():
    r = XrRuntime("panda3d-vr tests", debug=True)
    try:
        r.create_instance()
    except Exception as e:
        pytest.skip("no OpenXR runtime: %s" % e)
    yield r
    r.destroy()


def test_instance_and_extensions(rt):
    assert rt.instance is not None and rt.runtime_name
    assert xr.KHR_OPENGL_ENABLE_EXTENSION_NAME in rt.enabled_extensions
    logging.getLogger("panda3d_vr").info("enabled: %s", sorted(rt.enabled_extensions))


def test_every_binding_profile_is_valid(rt, base):
    class _VR:
        render = base.render
        tracking_space = base.render

    ctrls = [Controller(_VR, h, base.render) for h in ("left", "right")]
    inp = XrInput(rt, ctrls, custom_actions={
        "jump": {"kind": "bool", "bindings": {
            "/interaction_profiles/oculus/touch_controller": {"right": "input/a/click"}}}})
    inp.create()
    try:
        assert not inp.rejected_profiles, inp.rejected_profiles
        expected = set(PROFILES)
        if rt.has("hand_interaction"):
            expected.add(HAND_INTERACTION_PROFILE[0])
        assert expected <= set(inp.accepted_profiles)
        assert "jump" in inp.actions
    finally:
        inp.destroy()
        for c in ctrls:
            c.grip.remove_node()
            c.aim.remove_node()


def test_extension_entry_points_resolve(rt):
    for name, pfn, feature in (
        ("xrGetOpenGLGraphicsRequirementsKHR", xr.PFN_xrGetOpenGLGraphicsRequirementsKHR, None),
        ("xrCreateHandTrackerEXT", xr.PFN_xrCreateHandTrackerEXT, "hand_tracking"),
        ("xrGetVisibilityMaskKHR", xr.PFN_xrGetVisibilityMaskKHR, "visibility_mask"),
        ("xrRequestDisplayRefreshRateFB", xr.PFN_xrRequestDisplayRefreshRateFB, "refresh_rate"),
        ("xrCreatePassthroughFB", xr.PFN_xrCreatePassthroughFB, "passthrough"),
    ):
        if feature and not rt.has(feature):
            continue
        assert rt.fn(name, pfn)
