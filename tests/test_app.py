import pytest
from panda3d.core import LPoint3f

import panda3d_vr.core as core
from panda3d_vr import Interaction, Locomotion
from panda3d_vr.geometry import box


@pytest.fixture
def vr(base):
    v = core.VRManager(base, enabled=False, show_controllers=True)
    yield v
    base.ignore_all()
    v.destroy()


def _step(base, n=1):
    for _ in range(n):
        base.taskMgr.step()


def test_simulator_drives_rig(base, vr):
    assert vr.simulated and not vr.connected
    _step(base, 2)
    assert abs(vr.head.get_z() - vr.eye_height) < 1e-6
    assert vr.left.connected and vr.right.connected
    assert base.camera.get_parent() == vr.head
    # Hands sit in front of the head.
    assert vr.right.grip.get_pos(vr.head).y > 0


def test_button_events_edges(base, vr):
    got = []
    base.accept("vr-right-trigger", lambda c: got.append(("down", c.hand)))
    base.accept("vr-right-trigger-up", lambda c: got.append(("up", c.hand)))
    c = vr.right
    c._begin_frame()
    c._set_analog("trigger", 0.5)   # below press threshold
    assert got == []
    c._set_analog("trigger", 0.6)
    assert got == [("down", "right")] and c.was_pressed("trigger") and c.is_down("trigger")
    c._begin_frame()
    c._set_analog("trigger", 0.5)   # hysteresis: still held
    assert c.is_down("trigger") and not c.was_pressed("trigger")
    c._set_analog("trigger", 0.4)
    assert got[-1] == ("up", "right") and c.was_released("trigger")


def test_stick_directions(base, vr):
    got = []
    base.accept("vr-left-thumbstick_up", lambda c: got.append("up"))
    vr.left._set_stick(0.0, 0.9)
    vr.left._set_stick(0.0, 0.6)
    vr.left._set_stick(0.0, 0.3)
    vr.left._set_stick(0.0, 0.9)
    assert got == ["up", "up"]


def test_recenter(base, vr):
    vr.simulator.disable()
    vr.head.set_pos(0.7, -0.4, 1.7)
    vr.head.set_hpr(35, 10, 0)
    vr.recenter()
    head = vr.head.get_pos(vr.rig)
    assert abs(head.x) < 1e-5 and abs(head.y) < 1e-5 and abs(head.z - 1.7) < 1e-5
    assert abs(vr.head.get_h(vr.rig)) < 1e-4


def test_snap_turn_pivots_on_head(base, vr):
    loco = Locomotion(vr, teleport=False)
    try:
        vr.simulator.disable()
        vr.head.set_pos(0.5, 0.5, 1.6)
        before = vr.head.get_pos(vr.render)
        loco.rotate_around_head(45)
        after = vr.head.get_pos(vr.render)
        assert (before - after).length() < 1e-5
        assert abs(vr.rig.get_h() - 45) < 1e-4
    finally:
        loco.destroy()


def test_teleport_lands_feet_on_target(base, vr):
    loco = Locomotion(vr)
    try:
        vr.simulator.disable()
        vr.head.set_pos(0.3, 0.2, 1.6)
        loco.teleport_to(LPoint3f(5, 5, 0.5))
        head = vr.head.get_pos(vr.render)
        assert abs(head.x - 5) < 1e-5 and abs(head.y - 5) < 1e-5
        assert abs(vr.rig.get_z() - 0.5) < 1e-5
    finally:
        loco.destroy()


def test_teleport_arc_hits_floor(base, vr):
    loco = Locomotion(vr)
    try:
        vr.simulator.disable()
        vr.right.aim.set_pos(0, 0, 1.2)
        vr.right.aim.set_hpr(0, 20, 0)
        loco._update_arc()
        assert loco._target is not None and abs(loco._target.z) < 1e-4 and loco._target.y > 1
    finally:
        loco.destroy()


def test_grab_and_release(base, vr):
    inter = Interaction(vr, pointers=False)
    events = []
    base.accept("vr-grab", lambda np, c: events.append("grab"))
    base.accept("vr-release", lambda np, c, v: events.append("release"))
    cube = base.render.attach_new_node(box((0.05, 0.05, 0.05)))
    try:
        vr.simulator.disable()
        inter.make_grabbable(cube, 0.2)
        vr.right.grip.set_pos(0, 0.5, 1.0)
        cube.set_pos(0, 0.55, 1.0)
        c = vr.right
        c._begin_frame()
        c._set_analog("squeeze", 1.0)
        _step(base)
        assert inter.held[1] == cube and cube.get_parent() == c.grip
        world = cube.get_pos(base.render)
        assert abs(world.y - 0.55) < 1e-5, "grab keeps the world transform"
        c._begin_frame()
        c._set_analog("squeeze", 0.0)
        _step(base)
        assert inter.held[1] is None and cube.get_parent() == base.render
        assert events == ["grab", "release"]
    finally:
        inter.destroy()
        cube.remove_node()


def test_legacy_app_attributes():
    # BaseVrApp keeps the old attribute names working.
    for name in ("player", "vrCam", "hand_left", "hand_right", "cam_left", "cam_right",
                 "haptic_feedback", "reset_view_orientation", "UpdateHeadsetTracking"):
        assert hasattr(core.BaseVrApp, name)


def test_real_runtime_waits_for_headset(base):
    """With a runtime installed but no headset, the app keeps running in the simulator."""
    if not core.XR_AVAILABLE:
        pytest.skip("pyopenxr unavailable")
    vr = core.VRManager(base)
    try:
        _step(base, 3)
        if vr.connected:
            pytest.skip("a headset is connected")
        assert vr.status in ("waiting", "disconnected")
        assert vr.simulated
    finally:
        vr.destroy()
