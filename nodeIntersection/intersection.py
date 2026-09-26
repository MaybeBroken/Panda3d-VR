"""
Vectorised geometric intersection tests.

Two triangle meshes intersect exactly when some edge of one crosses a
triangle of the other, so mesh-mesh testing reduces to batched
segment/triangle tests (Moller-Trumbore), pruned with bounding boxes.
Coplanar touching triangles are not reported.
"""

import numpy as np

from .pandaToNumpy import geom_triangles, panda_mesh_to_numpy  # noqa: F401 (re-export)

_EPS = 1e-7
_CHUNK_PAIRS = 1 << 20  # edge/triangle pairs tested per batch (bounds memory)


def _as_triangles(mesh):
    if isinstance(mesh, np.ndarray):
        arr = mesh
    else:  # NodePath / GeomNode
        arr = geom_triangles(mesh)
    arr = np.asarray(arr, np.float64)
    if arr.ndim == 2:
        arr = arr[: len(arr) // 3 * 3].reshape(-1, 3, 3)
    return arr


def _aabb(tris):
    return tris.min(axis=1), tris.max(axis=1)


def _boxes_overlap(amin, amax, bmin, bmax):
    return np.all((amin <= bmax) & (bmin <= amax), axis=-1)


def _edges(tris):
    a = tris
    b = np.roll(tris, -1, axis=1)
    return a.reshape(-1, 3), b.reshape(-1, 3)


def segment_triangle_hits(p0, p1, tris):
    """
    Pairwise segment/triangle intersection for already-matched rows.
    Returns (hit_mask, points).
    """
    d = p1 - p0
    v0 = tris[:, 0]
    e1 = tris[:, 1] - v0
    e2 = tris[:, 2] - v0
    h = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(det) > _EPS
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = p0 - v0
    u = inv * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1)
    v = inv * np.einsum("ij,ij->i", d, q)
    t = inv * np.einsum("ij,ij->i", e2, q)
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)
    return hit, p0 + d * t[:, None]


def _edges_vs_triangles(src_tris, dst_tris, first_only):
    if not len(src_tris) or not len(dst_tris):
        return []
    dmin, dmax = _aabb(dst_tris)
    p0, p1 = _edges(src_tris)
    emin = np.minimum(p0, p1)
    emax = np.maximum(p0, p1)
    # Discard edges outside the other mesh's overall box.
    keep = _boxes_overlap(emin, emax, dmin.min(0), dmax.max(0))
    p0, p1, emin, emax = p0[keep], p1[keep], emin[keep], emax[keep]
    if not len(p0):
        return []
    points = []
    rows = max(1, _CHUNK_PAIRS // len(dst_tris))
    for start in range(0, len(p0), rows):
        sl = slice(start, start + rows)
        overlap = _boxes_overlap(emin[sl, None], emax[sl, None], dmin[None], dmax[None])
        ei, ti = np.nonzero(overlap)
        if not len(ei):
            continue
        hit, pts = segment_triangle_hits(p0[sl][ei], p1[sl][ei], dst_tris[ti])
        if hit.any():
            points.append(pts[hit])
            if first_only:
                break
    return points


def do_meshes_intersect(mesh1, mesh2):
    """
    True if two triangle meshes intersect.  Accepts (N,3,3) triangle arrays,
    (N,3) vertex arrays (consecutive triples) or Panda NodePaths/GeomNodes.
    """
    a = _as_triangles(mesh1)
    b = _as_triangles(mesh2)
    if not len(a) or not len(b):
        return False
    amin, amax = _aabb(a)
    bmin, bmax = _aabb(b)
    if not _boxes_overlap(amin.min(0), amax.max(0), bmin.min(0), bmax.max(0)):
        return False
    return bool(_edges_vs_triangles(a, b, True) or _edges_vs_triangles(b, a, True))


def compute_intersection_points(mesh1, mesh2):
    """(K, 3) array of points where edges of either mesh pierce the other."""
    a = _as_triangles(mesh1)
    b = _as_triangles(mesh2)
    pts = _edges_vs_triangles(a, b, False) + _edges_vs_triangles(b, a, False)
    if not pts:
        return np.zeros((0, 3))
    return np.concatenate(pts)


def sphere_pairs(centers_a, radii_a, centers_b, radii_b):
    """
    All overlapping (i, j) pairs between two sphere sets, as two index arrays.
    Uses a KD-tree for large sets when SciPy is available.
    """
    ca = np.asarray(centers_a, np.float64).reshape(-1, 3)
    cb = np.asarray(centers_b, np.float64).reshape(-1, 3)
    ra = np.asarray(radii_a, np.float64).reshape(-1)
    rb = np.asarray(radii_b, np.float64).reshape(-1)
    if not len(ca) or not len(cb):
        return np.zeros(0, int), np.zeros(0, int)
    if len(ca) * len(cb) > 4_000_000:
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            pass
        else:
            tree = cKDTree(cb)
            candidates = tree.query_ball_point(ca, ra + rb.max())
            ii = np.repeat(np.arange(len(ca)), [len(c) for c in candidates])
            jj = np.fromiter((j for c in candidates for j in c), int, len(ii))
            d2 = np.einsum("ij,ij->i", ca[ii] - cb[jj], ca[ii] - cb[jj])
            ok = d2 <= (ra[ii] + rb[jj]) ** 2
            return ii[ok], jj[ok]
    diff = ca[:, None, :] - cb[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    return np.nonzero(d2 <= (ra[:, None] + rb[None, :]) ** 2)
