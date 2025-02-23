# This is a module designed to detect intersections between basic meshes in 3D space.
# It provides a function to check if two meshes intersect and another function to compute the intersection points.
#

# Import necessary libraries
import numpy as np
from scipy.spatial import Delaunay
from .pandaToNumpy import panda_mesh_to_numpy


# Function to check if two meshes intersect
def do_meshes_intersect(mesh1, mesh2):
    """
    Check if two meshes intersect.

    Parameters:
    mesh1 (numpy.ndarray): First mesh represented as a 3D numpy array of shape (n, 3).
    mesh2 (numpy.ndarray): Second mesh represented as a 3D numpy array of shape (m, 3).

    Returns:
    bool: True if the meshes intersect, False otherwise.
    """
    # Check if there are enough points to create Delaunay triangulations
    if len(mesh1) < 5 or len(mesh2) < 5:
        raise ValueError("Not enough points to create Delaunay triangulations")

    # Create Delaunay triangulations for both meshes
    tri1 = Delaunay(mesh1, qhull_options='QJ')
    tri2 = Delaunay(mesh2, qhull_options='QJ')

    # Check if any triangle from mesh1 intersects with any triangle from mesh2
    for simplex1 in tri1.simplices:
        for simplex2 in tri2.simplices:
            if do_triangles_intersect(mesh1[simplex1], mesh2[simplex2]):
                return True
    return False


# Function to check if two triangles intersect
def do_triangles_intersect(triangle1, triangle2):
    """
    Check if two triangles intersect.

    Parameters:
    triangle1 (numpy.ndarray): First triangle represented as a 3D numpy array of shape (3, 3).
    triangle2 (numpy.ndarray): Second triangle represented as a 3D numpy array of shape (3, 3).

    Returns:
    bool: True if the triangles intersect, False otherwise.
    """
    # Compute the normal vectors of the triangles
    normal1 = np.cross(triangle1[1] - triangle1[0], triangle1[2] - triangle1[0])
    normal2 = np.cross(triangle2[1] - triangle2[0], triangle2[2] - triangle2[0])

    # Check if the planes of the triangles are parallel
    if np.abs(np.dot(normal1, normal2)) < 1e-6:
        return False

    # Compute the intersection line of the planes
    line_direction = np.cross(normal1, normal2)

    # Compute the intersection points
    intersection_points = []

    for i in range(3):
        for j in range(3):
            point = line_intersection(
                triangle1[i],
                triangle1[(i + 1) % 3],
                triangle2[j],
                triangle2[(j + 1) % 3],
            )
            if point is not None:
                intersection_points.append(point)

    # Check if any of the intersection points are inside both triangles
    for point in intersection_points:
        if is_point_in_triangle(point, triangle1) and is_point_in_triangle(
            point, triangle2
        ):
            return True

    return False


# Function to compute the intersection point of two line segments
def line_intersection(p1, p2, q1, q2):
    """
    Compute the intersection point of two line segments.

    Parameters:
    p1 (numpy.ndarray): First point of the first line segment.
    p2 (numpy.ndarray): Second point of the first line segment.
    q1 (numpy.ndarray): First point of the second line segment.
    q2 (numpy.ndarray): Second point of the second line segment.

    Returns:
    numpy.ndarray: Intersection point if it exists, None otherwise.
    """
    # Compute the direction vectors
    d1 = p2 - p1
    d2 = q2 - q1

    # Compute the cross product
    cross = np.cross(d1, d2)

    # Check if the lines are parallel
    if np.linalg.norm(cross) < 1e-6:
        return None

    # Compute the intersection point
    denominator = np.dot(d1, cross)
    if denominator == 0:
        return None
    t = np.dot(q1 - p1, cross) / denominator

    if 0 <= t <= 1:
        return p1 + t * d1

    return None


# Function to check if a point is inside a triangle
def is_point_in_triangle(point, triangle):
    """
    Check if a point is inside a triangle.

    Parameters:
    point (numpy.ndarray): Point to check.
    triangle (numpy.ndarray): Triangle represented as a 3D numpy array of shape (3, 3).

    Returns:
    bool: True if the point is inside the triangle, False otherwise.
    """
    # Compute the barycentric coordinates
    v0 = triangle[1] - triangle[0]
    v1 = triangle[2] - triangle[0]
    v2 = point - triangle[0]

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    if denom == 0:
        return False

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom

    return (v >= 0) and (w >= 0) and (v + w <= 1)


# Function to compute the intersection points of two meshes
def compute_intersection_points(mesh1, mesh2):
    """
    Compute the intersection points of two meshes.

    Parameters:
    mesh1 (numpy.ndarray): First mesh represented as a 3D numpy array of shape (n, 3).
    mesh2 (numpy.ndarray): Second mesh represented as a 3D numpy array of shape (m, 3).

    Returns:
    list: List of intersection points.
    """
    # Create Delaunay triangulations for both meshes
    tri1 = Delaunay(mesh1)
    tri2 = Delaunay(mesh2)

    # Initialize an empty list to store the intersection points
    intersection_points = []

    # Check if any triangle from mesh1 intersects with any triangle from mesh2
    for simplex1 in tri1.simplices:
        for simplex2 in tri2.simplices:
            point = do_triangles_intersect(mesh1[simplex1], mesh2[simplex2])
            if point is not None:
                intersection_points.append(point)

    return intersection_points
