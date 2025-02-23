from panda3d.core import (
    GeomVertexReader,
    GeomVertexWriter,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexFormat,
    GeomVertexData,
)
import numpy as np


def panda_mesh_to_numpy(geom_node):
    """
    Convert a Panda3D GeomNode to a numpy array.

    Parameters:
    geom_node (GeomNode): The Panda3D GeomNode to convert.

    Returns:
    numpy.ndarray: The mesh represented as a 3D numpy array of shape (n, 3).
    """
    vertices = []

    for geom in geom_node.get_geoms():
        vdata = geom.get_vertex_data()
        reader = GeomVertexReader(vdata, "vertex")

        while not reader.is_at_end():
            vertex = reader.get_data3f()
            vertices.append([vertex[0], vertex[1], vertex[2]])

    return np.array(vertices)


def numpy_array_to_mesh(numpy_array):
    """
    Convert a numpy array to a Panda3D GeomNode.

    Parameters:
    numpy_array (numpy.ndarray): The numpy array to convert.

    Returns:
    GeomNode: The converted Panda3D GeomNode.
    """

    # Create a vertex format
    format = GeomVertexFormat.get_v3()
    vdata = GeomVertexData("vertices", format, Geom.UH_static)

    # Create a vertex writer
    vertex_writer = GeomVertexWriter(vdata, "vertex")

    # Write the vertices to the vertex data
    for vertex in numpy_array:
        vertex_writer.add_data3f(vertex[0], vertex[1], vertex[2])

    # Create a geom
    geom = Geom(vdata)

    # Create triangles
    triangles = GeomTriangles(Geom.UH_static)

    # Add triangles (assuming the numpy array is structured as triangles)
    for i in range(0, len(numpy_array), 3):
        if i + 2 < len(numpy_array):
            triangles.add_vertices(i, i + 1, i + 2)

    geom.add_primitive(triangles)

    # Create a geom node
    geom_node = GeomNode("mesh")
    geom_node.add_geom(geom)

    return geom_node
