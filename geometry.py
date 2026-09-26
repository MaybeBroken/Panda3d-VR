"""
Procedural geometry built straight from numpy arrays.

Vertex data is written through Panda's buffer protocol in one memcpy per
column instead of millions of GeomVertexWriter calls.
"""

import numpy as np
from panda3d.core import (
    Geom,
    GeomLines,
    GeomNode,
    GeomTriangles,
    GeomVertexArrayFormat,
    GeomVertexData,
    GeomVertexFormat,
    InternalName,
)

__all__ = [
    "make_geom",
    "make_geom_node",
    "uv_sphere",
    "box_from_corners",
    "box",
    "line_node",
]

_FORMATS = {}


def _format(has_normal, has_color, has_uv):
    key = (has_normal, has_color, has_uv)
    fmt = _FORMATS.get(key)
    if fmt is None:
        f = GeomVertexFormat()
        columns = [(InternalName.get_vertex(), 3, Geom.C_point)]
        if has_normal:
            columns.append((InternalName.get_normal(), 3, Geom.C_normal))
        if has_color:
            columns.append((InternalName.get_color(), 4, Geom.C_color))
        if has_uv:
            columns.append((InternalName.get_texcoord(), 2, Geom.C_texcoord))
        # One tightly packed array per column so each can be memcpy'd directly.
        for name, n, contents in columns:
            af = GeomVertexArrayFormat()
            af.add_column(name, n, Geom.NT_float32, contents)
            f.add_array(af)
        fmt = _FORMATS[key] = GeomVertexFormat.register_format(f)
    return fmt


def _write(array_data, values):
    buf = np.ascontiguousarray(values, dtype=np.float32)
    memoryview(array_data).cast("B")[:] = buf.tobytes()


def make_geom(vertices, indices, normals=None, colors=None, uvs=None, lines=False, usage=Geom.UH_static):
    vertices = np.asarray(vertices, np.float32).reshape(-1, 3)
    n = len(vertices)
    fmt = _format(normals is not None, colors is not None, uvs is not None)
    vdata = GeomVertexData("mesh", fmt, usage)
    vdata.unclean_set_num_rows(n)
    arrays = [vertices]
    if normals is not None:
        arrays.append(np.broadcast_to(np.asarray(normals, np.float32), (n, 3)))
    if colors is not None:
        arrays.append(np.broadcast_to(np.asarray(colors, np.float32), (n, 4)))
    if uvs is not None:
        arrays.append(np.asarray(uvs, np.float32).reshape(-1, 2))
    for i, a in enumerate(arrays):
        _write(vdata.modify_array(i), a)

    prim = (GeomLines if lines else GeomTriangles)(usage)
    prim.set_index_type(Geom.NT_uint32)
    idx = np.ascontiguousarray(indices, dtype=np.uint32).ravel()
    iarr = prim.modify_vertices()
    iarr.unclean_set_num_rows(len(idx))
    memoryview(iarr).cast("B")[:] = idx.tobytes()
    geom = Geom(vdata)
    geom.add_primitive(prim)
    return geom


def make_geom_node(name, vertices, indices, **kw):
    node = GeomNode(name)
    node.add_geom(make_geom(vertices, indices, **kw))
    return node


def uv_sphere(radius=1.0, lat=16, lon=24, color=(1, 1, 1, 1), name="sphere"):
    """UV sphere GeomNode (vectorised; replaces the old per-vertex Python loop)."""
    i = np.arange(lat + 1, dtype=np.float32)[:, None]
    j = np.arange(lon + 1, dtype=np.float32)[None, :]
    theta = np.pi * i / lat
    phi = 2 * np.pi * j / lon
    n = np.stack(
        np.broadcast_arrays(np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)),
        axis=-1,
    ).reshape(-1, 3)
    uv = np.stack(np.broadcast_arrays(j / lon, i / lat), axis=-1).reshape(-1, 2)
    a = (np.arange(lat)[:, None] * (lon + 1) + np.arange(lon)[None, :]).ravel()
    b = a + lon + 1
    # Counter-clockwise seen from outside, so Panda's back-face culling keeps them.
    tris = np.stack([a, b, b + 1, a, b + 1, a + 1], axis=-1).reshape(-1, 2, 3)
    # Drop the zero-area triangles that touch the poles.
    row = np.repeat(np.arange(lat), lon)
    keep = np.ones((len(tris), 2), bool)
    keep[row == 0, 1] = False
    keep[row == lat - 1, 0] = False
    tris = tris[keep]
    return make_geom_node(name, n * radius, tris, normals=n, colors=color, uvs=uv)


_BOX_FACES = np.array(
    [
        [0, 3, 2, 1],  # -z
        [4, 5, 6, 7],  # +z
        [0, 1, 5, 4],  # -y
        [2, 3, 7, 6],  # +y
        [0, 4, 7, 3],  # -x
        [1, 2, 6, 5],  # +x
    ]
)


def box_from_corners(points, colors=None, name="cube"):
    """
    Box from 8 corners ordered like the old ``Cube`` helper (bottom 0-3, top 4-7).
    Emits 24 vertices so every face gets a proper flat normal.
    """
    pts = np.asarray(points, np.float32).reshape(8, 3)
    cols = None if colors is None else np.broadcast_to(np.asarray(colors, np.float32), (8, 4))
    center = pts.mean(axis=0)
    verts, norms, vcols, uvs, tris = [], [], [], [], []
    quad_uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    for face in _BOX_FACES:
        q = pts[face]
        nrm = np.cross(q[1] - q[0], q[2] - q[0])
        if np.dot(nrm, q.mean(axis=0) - center) < 0:  # enforce outward winding
            face = face[::-1]
            q = pts[face]
            nrm = -nrm
        length = np.linalg.norm(nrm)
        nrm = nrm / length if length > 0 else nrm
        base = len(verts) * 4
        verts.append(q)
        norms.append(np.repeat(nrm[None], 4, 0))
        if cols is not None:
            vcols.append(cols[face])
        uvs.append(quad_uv)
        tris.append([base, base + 1, base + 2, base, base + 2, base + 3])
    return make_geom_node(
        name,
        np.concatenate(verts),
        np.array(tris),
        normals=np.concatenate(norms),
        colors=np.concatenate(vcols) if vcols else None,
        uvs=np.concatenate(uvs),
    )


def box(half_extents=(0.5, 0.5, 0.5), center=(0, 0, 0), color=(1, 1, 1, 1), name="box"):
    hx, hy, hz = half_extents
    cx, cy, cz = center
    corners = [
        (cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz), (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz),
    ]
    return box_from_corners(corners, color, name)


def line_node(points, color=(1, 1, 1, 1), name="line"):
    pts = np.asarray(points, np.float32).reshape(-1, 3)
    idx = np.stack([np.arange(len(pts) - 1), np.arange(1, len(pts))], axis=-1)
    return make_geom_node(name, pts, idx, colors=color, lines=True)
