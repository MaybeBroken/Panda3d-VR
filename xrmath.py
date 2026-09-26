"""
Conversions between OpenXR and Panda3D coordinate systems.

OpenXR: right handed, +Y up, -Z forward, metres.
Panda3D: right handed, +Z up, +Y forward, arbitrary units (``scale`` per metre).

The mapping is a pure rotation of the basis, (x, y, z)_xr -> (x, -z, y)_panda,
so quaternions convert component-wise and no Euler angles are ever involved
(Euler round-trips were the source of the old "looking straight up breaks
everything" bug).
"""

import math

from panda3d.core import LPoint3f, LQuaternionf, LVector3f

import xr

__all__ = [
    "apply_pose",
    "pose_to_panda",
    "panda_to_pose",
    "vector_to_panda",
    "fov_to_film",
    "xr_time_to_seconds",
]


def apply_pose(node, pose, scale=1.0):
    """Set a NodePath's pos/quat from an ``xr.Posef`` (hot path, no temporaries)."""
    p = pose.position
    o = pose.orientation
    node.set_pos_quat(
        LPoint3f(p.x * scale, -p.z * scale, p.y * scale),
        LQuaternionf(o.w, o.x, -o.z, o.y),
    )


def pose_to_panda(pose, scale=1.0):
    p = pose.position
    o = pose.orientation
    return (
        LPoint3f(p.x * scale, -p.z * scale, p.y * scale),
        LQuaternionf(o.w, o.x, -o.z, o.y),
    )


def panda_to_pose(pos, quat, scale=1.0, out=None):
    """Convert a Panda pos/quat (in tracking-space units) back into an ``xr.Posef``."""
    inv = 1.0 / scale
    if out is None:
        out = xr.Posef()
    out.position.x = pos[0] * inv
    out.position.y = pos[2] * inv
    out.position.z = -pos[1] * inv
    # LQuaternionf is (r, i, j, k) == (w, x, y, z)
    out.orientation.w = quat[0]
    out.orientation.x = quat[1]
    out.orientation.y = quat[3]
    out.orientation.z = -quat[2]
    return out


def vector_to_panda(v, scale=1.0):
    return LVector3f(v.x * scale, -v.z * scale, v.y * scale)


def fov_to_film(fov):
    """
    Asymmetric OpenXR ``xr.Fovf`` -> (film_w, film_h, offset_x, offset_y) for a
    Panda PerspectiveLens with focal length 1.  This gives each eye its exact
    off-axis frustum instead of a symmetric lens plus an image-shift hack.
    """
    tl = math.tan(fov.angle_left)
    tr = math.tan(fov.angle_right)
    tu = math.tan(fov.angle_up)
    td = math.tan(fov.angle_down)
    return tr - tl, tu - td, (tr + tl) * 0.5, (tu + td) * 0.5


def xr_time_to_seconds(t):
    return t * 1e-9
