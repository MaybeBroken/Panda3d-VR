import numpy as np
from panda3d.core import NodePath

from panda3d_vr.geometry import box
from panda3d_vr.nodeIntersection import CollisionWorld, Cube, Sphere
from panda3d_vr.nodeIntersection.intersection import (
    compute_intersection_points,
    do_meshes_intersect,
    sphere_pairs,
)
from panda3d_vr.nodeIntersection.pandaToNumpy import geom_triangles


def _box(center, half=0.5):
    return geom_triangles(NodePath(box((half,) * 3, center)))


def test_mesh_overlap_detected():
    assert do_meshes_intersect(_box((0, 0, 0)), _box((0.7, 0.2, 0.1)))


def test_mesh_separated():
    assert not do_meshes_intersect(_box((0, 0, 0)), _box((1.5, 0, 0)))


def test_mesh_points_on_both_surfaces():
    pts = compute_intersection_points(_box((0, 0, 0)), _box((0.7, 0.2, 0.1)))
    assert len(pts)
    # Every point lies inside (or on) both boxes.
    assert (np.abs(pts) <= 0.5 + 1e-6).all()
    assert (np.abs(pts - (0.7, 0.2, 0.1)) <= 0.5 + 1e-6).all()


def test_nodepaths_accepted():
    a = NodePath(box((0.5, 0.5, 0.5)))
    b = NodePath(box((0.5, 0.5, 0.5), (0.9, 0, 0)))
    assert do_meshes_intersect(a, b)


def test_sphere_pairs_matches_bruteforce():
    rng = np.random.default_rng(1)
    a, b = rng.random((50, 3)) * 5, rng.random((70, 3)) * 5
    ra, rb = rng.random(50) * 0.5, rng.random(70) * 0.5
    ii, jj = sphere_pairs(a, ra, b, rb)
    got = set(zip(ii.tolist(), jj.tolist()))
    want = {(i, j) for i in range(50) for j in range(70)
            if np.linalg.norm(a[i] - b[j]) <= ra[i] + rb[j]}
    assert got == want


def test_world_enter_exit_events(base):
    world = CollisionWorld()
    hand = base.render.attach_new_node("hand")
    actor = world.add_sphere("hand", 0.1, node=hand)
    world.add_base_collider(0.2, (1, 0, 0), "button")
    events = []
    base.accept("collision-enter", lambda r: events.append(("enter", r.colliderStr)))
    base.accept("collision-exit", lambda r: events.append(("exit", r.colliderStr)))
    world.update()
    hand.set_pos(0.95, 0, 0)
    world.update()
    assert actor.collision_report and actor.collision_report[0].colliderStr == "button"
    world.update()  # still touching: no duplicate enter
    hand.set_pos(5, 0, 0)
    world.update()
    assert events == [("enter", "button"), ("exit", "button")]
    base.ignore_all()
    hand.remove_node()


def test_mesh_bodies_follow_nodes(base):
    world = CollisionWorld()
    a = base.render.attach_new_node(box((0.5, 0.5, 0.5)))
    b = base.render.attach_new_node(box((0.5, 0.5, 0.5)))
    b.set_pos(3, 0, 0)
    world.add_mesh("a", a, kind="actor")
    world.add_mesh("b", b, kind="collider")
    assert world.update() == []
    b.set_pos(0.8, 0, 0)
    reports = world.update()
    assert len(reports) == 1 and len(reports[0].points)
    a.remove_node()
    b.remove_node()


def test_legacy_generators():
    assert Sphere(1.0, 8, 8).get_num_geoms() == 1
    pts = [[x, y, z, (1, 0, 0, 1)] for z in (0, 1) for (x, y) in ((0, 0), (1, 0), (1, 1), (0, 1))]
    assert Cube(pts).get_geom(0).get_vertex_data().get_num_rows() == 24
