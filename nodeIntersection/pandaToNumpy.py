"""
Fast conversion between Panda3D geometry and numpy arrays.

Vertex and index buffers are read through Panda's buffer protocol, so a
100k-triangle model converts in milliseconds instead of seconds.
"""

import numpy as np
from panda3d.core import Geom, GeomNode, GeomVertexFormat, NodePath

from ..geometry import make_geom_node

_INDEX_DTYPES = {Geom.NT_uint8: np.uint8, Geom.NT_uint16: np.uint16, Geom.NT_uint32: np.uint32}


def _geom_nodes(obj):
    if isinstance(obj, GeomNode):
        yield obj, None
        return
    np_ = obj if isinstance(obj, NodePath) else NodePath(obj)
    if np_.node().is_geom_node():
        yield np_.node(), np_
    for child in np_.find_all_matches("**/+GeomNode"):
        yield child.node(), child


def _positions(vdata):
    vdata = vdata.convert_to(GeomVertexFormat.get_v3())
    return np.frombuffer(memoryview(vdata.get_array(0)), dtype=np.float32).reshape(-1, 3)


def _to_numpy_mat(mat):
    return np.array(mat, dtype=np.float32).reshape(4, 4)


def geom_triangles(obj, relative_to=None):
    """
    All triangles under a NodePath/GeomNode as a float32 array of shape (N, 3, 3).

    With ``relative_to`` (a NodePath) the triangles are transformed into that
    node's coordinate space, e.g. ``render`` for world space.
    """
    out = []
    for gnode, gnp in _geom_nodes(obj):
        mat = None
        if relative_to is not None and gnp is not None:
            mat = _to_numpy_mat(gnp.get_mat(relative_to))
        for gi in range(gnode.get_num_geoms()):
            geom = gnode.get_geom(gi).decompose()
            pos = _positions(geom.get_vertex_data())
            if mat is not None:
                pos = pos @ mat[:3, :3] + mat[3, :3]
            for pi in range(geom.get_num_primitives()):
                prim = geom.get_primitive(pi)
                if prim.get_num_vertices_per_primitive() != 3:
                    continue
                if prim.is_indexed():
                    idx = np.frombuffer(memoryview(prim.get_vertices()),
                                        dtype=_INDEX_DTYPES[prim.get_index_type()])
                else:
                    first = prim.get_first_vertex()
                    idx = np.arange(first, first + prim.get_num_vertices())
                idx = idx[: len(idx) // 3 * 3].astype(np.int64)
                out.append(pos[idx].reshape(-1, 3, 3))
    if not out:
        return np.zeros((0, 3, 3), np.float32)
    return np.ascontiguousarray(np.concatenate(out), dtype=np.float32)


def panda_mesh_to_numpy(obj, relative_to=None):
    """All vertex positions under ``obj`` as an (N, 3) float32 array."""
    out = []
    for gnode, gnp in _geom_nodes(obj):
        mat = None
        if relative_to is not None and gnp is not None:
            mat = _to_numpy_mat(gnp.get_mat(relative_to))
        for gi in range(gnode.get_num_geoms()):
            pos = _positions(gnode.get_geom(gi).get_vertex_data())
            if mat is not None:
                pos = pos @ mat[:3, :3] + mat[3, :3]
            out.append(pos)
    if not out:
        return np.zeros((0, 3), np.float32)
    return np.concatenate(out)


def numpy_array_to_mesh(numpy_array, name="mesh"):
    """
    Build a GeomNode from either an (N, 3, 3) triangle array or an (N, 3)
    vertex array where every 3 consecutive vertices form a triangle.
    """
    verts = np.asarray(numpy_array, np.float32).reshape(-1, 3)
    n = len(verts) // 3 * 3
    verts = verts[:n]
    return make_geom_node(name, verts, np.arange(n, dtype=np.uint32).reshape(-1, 3))
