import math

import numpy as np
import pytest
from panda3d.core import LVector3f, NodePath, PerspectiveLens, LVecBase4f, Point3

xr = pytest.importorskip("xr")

from panda3d_vr.geometry import box, uv_sphere  # noqa: E402
from panda3d_vr.nodeIntersection.pandaToNumpy import geom_triangles  # noqa: E402
from panda3d_vr.xrmath import apply_pose, fov_to_film, panda_to_pose, pose_to_panda  # noqa: E402


def _pose(q=(0, 0, 0, 1), p=(0, 0, 0)):
    return xr.Posef(orientation=xr.Quaternionf(*q), position=xr.Vector3f(*p))


def test_identity_pose_looks_forward():
    np_ = NodePath("n")
    apply_pose(np_, _pose(p=(1, 2, 3)))
    assert np_.get_pos().almost_equal(Point3(1, -3, 2))
    fwd = np_.get_quat().get_forward()
    assert fwd.almost_equal(LVector3f(0, 1, 0))


def test_yaw_left_maps_to_heading():
    s = math.sin(math.radians(45))
    np_ = NodePath("n")
    apply_pose(np_, _pose(q=(0, s, 0, s)))  # +90 deg about XR +Y (turn left)
    assert np_.get_quat().get_forward().almost_equal(LVector3f(-1, 0, 0), 1e-5)
    assert abs(np_.get_h() - 90) < 1e-3


def test_pitch_up_straight_is_stable():
    # The old Euler hack broke when looking straight up.
    s = math.sin(math.radians(45))
    np_ = NodePath("n")
    apply_pose(np_, _pose(q=(s, 0, 0, s)))  # +90 deg about XR +X (look up)
    assert np_.get_quat().get_forward().almost_equal(LVector3f(0, 0, 1), 1e-5)


def test_pose_roundtrip():
    q = np.random.randn(4)
    q /= np.linalg.norm(q)
    p = _pose(q=tuple(q), p=(0.3, 1.2, -0.7))
    pos, quat = pose_to_panda(p, 2.0)
    back = panda_to_pose(pos, quat, 2.0)
    for a, b in zip((back.position.x, back.position.y, back.position.z), (0.3, 1.2, -0.7)):
        assert abs(a - b) < 1e-5
    for a, b in zip((back.orientation.x, back.orientation.y, back.orientation.z, back.orientation.w), q):
        assert abs(a - b) < 1e-5


def test_asymmetric_frustum_edges():
    fov = xr.Fovf(angle_left=-0.8, angle_right=0.7, angle_up=0.75, angle_down=-0.9)
    w, h, ox, oy = fov_to_film(fov)
    lens = PerspectiveLens()
    lens.set_film_size(w, h)
    lens.set_film_offset(ox, oy)
    lens.set_focal_length(1.0)
    mat = lens.get_projection_mat()
    for ang, horizontal, expect in ((-0.8, True, -1), (0.7, True, 1), (0.75, False, 1), (-0.9, False, -1)):
        p = LVecBase4f(math.tan(ang), 1, 0, 1) if horizontal else LVecBase4f(0, 1, math.tan(ang), 1)
        out = mat.xform(p)
        v = out[0] / out[3] if horizontal else out[1] / out[3]
        assert abs(v - expect) < 1e-4


def _outward(tris):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    c = tris.mean(axis=1)
    return np.einsum("ij,ij->i", n, c)


def test_sphere_winding_outward():
    tris = geom_triangles(NodePath(uv_sphere(1.0, 12, 16)))
    area = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    d = _outward(tris)[area > 1e-8]
    assert (d > 0).all()


def test_box_winding_and_transform():
    np_ = NodePath(box((1, 2, 3)))
    tris = geom_triangles(np_)
    assert tris.shape == (12, 3, 3)
    assert (_outward(tris) > 0).all()
    root = NodePath("root")
    np_.reparent_to(root)
    np_.set_pos(10, 0, 0)
    world = geom_triangles(np_, relative_to=root)
    assert abs(world[..., 0].mean() - 10) < 1e-5
