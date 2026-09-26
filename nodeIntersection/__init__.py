"""
Lightweight collision reporting between "actors" (things that move, like
hands) and "colliders" (things they touch).

Bodies are spheres (fast, vectorised) or triangle meshes (exact).  The world
is updated from a Panda task, not a busy-looping thread, and fires
``collision-enter`` / ``collision-exit`` events with a CollisionReport.

The original names (``Mgr``, ``BaseActor``, ``add_base_actor``...) still work.
"""

import builtins
import random
import threading
import time

import numpy as np
from panda3d.core import NodePath

from ..geometry import box_from_corners, uv_sphere
from .intersection import (
    compute_intersection_points,
    do_meshes_intersect,
    segment_triangle_hits,
    sphere_pairs,
)
from .pandaToNumpy import geom_triangles, numpy_array_to_mesh, panda_mesh_to_numpy

__all__ = [
    "Body", "BaseActor", "BaseCollider", "ComplexActor", "ComplexCollider",
    "CollisionReport", "CollisionWorld", "Mgr", "Sphere", "Cube", "CubeGenerator",
    "create_uv_sphere", "create_cube", "getTotalDistance", "do_meshes_intersect",
    "compute_intersection_points", "panda_mesh_to_numpy", "numpy_array_to_mesh",
    "geom_triangles", "sphere_pairs", "segment_triangle_hits",
]


def _render():
    base = getattr(builtins, "base", None)
    return base.render if base is not None else None


# ------------------------------------------------------------------ geometry

def Sphere(radius, lat, lon):
    """UV sphere GeomNode."""
    return uv_sphere(radius, lat, lon)


def Cube(pointArray):
    """Box GeomNode from 8 points ``[[x, y, z, (r, g, b, a)], ...]``."""
    pts = [p[:3] for p in pointArray]
    cols = [p[3] for p in pointArray]
    return box_from_corners(pts, cols)


def create_uv_sphere(radius, resolution=(30, 30)):
    return NodePath(Sphere(radius, resolution[0], resolution[1]))


def create_cube(pointArray):
    return NodePath(Cube(pointArray))


def _corners(position, radius):
    x, y, z = position
    r = radius
    return [
        (x - r, y - r, z - r), (x + r, y - r, z - r), (x + r, y + r, z - r), (x - r, y + r, z - r),
        (x - r, y - r, z + r), (x + r, y - r, z + r), (x + r, y + r, z + r), (x - r, y + r, z + r),
    ]


class CubeGenerator:
    _BASE_COLORS = [
        (1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (1, 1, 0, 1),
        (1, 0.5, 0.5, 1), (0.5, 1, 0.5, 1), (0.5, 0.5, 1, 1), (1, 1, 0.5, 1),
    ]

    def raw(self, position, radius, color):
        return box_from_corners(_corners(position, radius), color)

    def base(self, position, radius):
        return NodePath(box_from_corners(_corners(position, radius), self._BASE_COLORS))

    def randomColor(self):
        cols = [(random.random(), random.random(), random.random(), 1) for _ in range(8)]
        return NodePath(box_from_corners(_corners((0.5, 0.5, -0.5), 0.5), cols))

    def randomShape(self):
        pts = np.random.rand(8, 3)
        cols = [(random.random(), random.random(), random.random(), 1) for _ in range(8)]
        return NodePath(box_from_corners(pts, cols))


def getTotalDistance(actor, collider):
    return float(np.linalg.norm(np.subtract(actor.position, collider.position)))


# -------------------------------------------------------------------- bodies

class Body:
    """
    A collision body.  ``shape`` is "sphere" or "mesh"; ``kind`` is "actor" or
    "collider".  If ``nodePath`` is set the body follows it every update.
    """

    def __init__(self, name, kind, shape, radius=0.0, position=(0, 0, 0), mesh=None, nodePath=None):
        self.name = name
        self.kind = kind
        self.shape = shape
        self.radius = float(radius)
        self.position = tuple(position)
        self.mesh = mesh
        self.nodePath = nodePath
        self.collision_report = None
        self._sphere = None
        self._local_tris = None
        if shape == "mesh":
            src = mesh if mesh is not None else nodePath
            self.nodePath = nodePath if nodePath is not None else (src if isinstance(src, NodePath) else None)
            self._local_tris = geom_triangles(src).astype(np.float64)

    @property
    def sphere(self):
        """Debug sphere NodePath (created on first use)."""
        if self._sphere is None:
            self._sphere = create_uv_sphere(self.radius or 0.1, (12, 16))
            self._sphere.set_render_mode_wireframe()
            self._sphere.set_light_off(1)
            self._sphere.set_pos(*self.position)
        return self._sphere

    @property
    def array(self):
        return self.world_triangles()

    def world_triangles(self):
        tris = self._local_tris
        if self.nodePath is None or self.nodePath.is_empty():
            return tris
        ref = _render() or self.nodePath.get_top()
        m = np.array(self.nodePath.get_mat(ref), np.float64).reshape(4, 4)
        return tris @ m[:3, :3] + m[3, :3]

    def _sync(self, ref):
        if self.nodePath is not None and not self.nodePath.is_empty() and ref is not None:
            p = self.nodePath.get_pos(ref)
            self.position = (p[0], p[1], p[2])
            if self._sphere is not None:
                self._sphere.set_pos(p)

    def __repr__(self):
        return "<%s %s %r>" % (self.kind, self.shape, self.name)


class BaseActor(Body):
    def __init__(self, radius, position, name, mesh=None, nodePath=None):
        super().__init__(name, "actor", "sphere", radius, position, mesh, nodePath)


class BaseCollider(Body):
    def __init__(self, radius, position, name, mesh=None, nodePath=None):
        super().__init__(name, "collider", "sphere", radius, position, mesh, nodePath)


class ComplexActor(Body):
    def __init__(self, mesh, name):
        super().__init__(name, "actor", "mesh", mesh=mesh)


class ComplexCollider(Body):
    def __init__(self, mesh, name):
        super().__init__(name, "collider", "mesh", mesh=mesh)


class CollisionReport:
    def __init__(self, actor, collider, actor_position, collider_position, points=None):
        self.actor = actor
        self.collider = collider
        self.actorStr = actor.name
        self.colliderStr = collider.name
        self.actor_position = actor_position
        self.collider_position = collider_position
        self.points = points

    @property
    def report(self):
        return {
            "actor": self.actor,
            "collider": self.collider,
            "actor_position": self.actor_position,
            "collider_position": self.collider_position,
        }

    def __repr__(self):
        return "CollisionReport(actor=%s, collider=%s, actor_position=%s, collider_position=%s)" % (
            self.actorStr, self.colliderStr, self.actor_position, self.collider_position)

    __str__ = __repr__


# --------------------------------------------------------------------- world

class CollisionWorld:
    def __init__(self):
        self.actors = []
        self.colliders = []
        self.reportedCollisions = []
        self._touching = {}
        self._task = None
        self._thread = None
        self._stop = False

    # ---- legacy views
    @property
    def base_actors(self):
        return [b for b in self.actors if b.shape == "sphere"]

    @property
    def complex_actors(self):
        return [b for b in self.actors if b.shape == "mesh"]

    @property
    def base_colliders(self):
        return [b for b in self.colliders if b.shape == "sphere"]

    @property
    def complex_colliders(self):
        return [b for b in self.colliders if b.shape == "mesh"]

    # ---- registration
    def add(self, body):
        (self.actors if body.kind == "actor" else self.colliders).append(body)
        return body

    def remove(self, body):
        for lst in (self.actors, self.colliders):
            if body in lst:
                lst.remove(body)
        for key in [k for k in self._touching if body in k]:
            del self._touching[key]
        return body

    def add_sphere(self, name, radius, node=None, position=(0, 0, 0), kind="actor"):
        return self.add(Body(name, kind, "sphere", radius, position, nodePath=node))

    def add_mesh(self, name, node, kind="collider"):
        return self.add(Body(name, kind, "mesh", mesh=node, nodePath=node))

    def add_base_actor(self, radius, position, name, mesh=None, nodePath=None):
        return self.add(BaseActor(radius, position, name, mesh, nodePath))

    def add_base_collider(self, radius, position, name, mesh=None, nodePath=None):
        return self.add(BaseCollider(radius, position, name, mesh, nodePath))

    def add_complex_actor(self, name, mesh):
        return self.add(ComplexActor(mesh, name))

    def add_complex_collider(self, mesh, name):
        return self.add(ComplexCollider(mesh, name))

    remove_base_actor = remove_complex_actor = remove
    remove_base_collider = remove_complex_collider = remove

    def setActorPosition(self, actor, position):
        actor.position = tuple(position)
        return actor

    setColliderPosition = setActorPosition

    def setActorMesh(self, actor, mesh):
        actor.mesh = mesh
        if actor.shape == "mesh":
            actor._local_tris = geom_triangles(mesh).astype(np.float64)
        return actor

    setColliderMesh = setActorMesh

    def transformActorType(self, actor):
        self.remove(actor)
        if actor.shape == "sphere":
            new = ComplexActor(actor.mesh, actor.name)
        else:
            ref = _render()
            pos = actor.nodePath.get_pos(ref) if actor.nodePath is not None and ref is not None else actor.position
            new = BaseActor(1.0, tuple(pos), actor.name, actor.mesh, actor.nodePath)
        return self.add(new)

    def clear(self):
        self.actors.clear()
        self.colliders.clear()
        self.reportedCollisions.clear()
        self._touching.clear()

    def get_reported_collisions(self):
        return self.reportedCollisions

    def showCollisions(self, parent=None):
        parent = parent or _render()
        for b in self.actors + self.colliders:
            if b.shape == "sphere" and parent is not None:
                b.sphere.reparent_to(parent)
                b.sphere.show()

    def hideCollisions(self):
        for b in self.actors + self.colliders:
            if b._sphere is not None:
                b._sphere.hide()

    # ---- simulation
    def update(self):
        ref = _render()
        for b in self.actors:
            b._sync(ref)
            b.collision_report = None
        for b in self.colliders:
            b._sync(ref)
            b.collision_report = None
        reports = []

        sa = [b for b in self.actors if b.shape == "sphere"]
        sc = [b for b in self.colliders if b.shape == "sphere"]
        if sa and sc:
            ii, jj = sphere_pairs([b.position for b in sa], [b.radius for b in sa],
                                  [b.position for b in sc], [b.radius for b in sc])
            for i, j in zip(ii.tolist(), jj.tolist()):
                reports.append(CollisionReport(sa[i], sc[j], sa[i].position, sc[j].position))

        ma = [b for b in self.actors if b.shape == "mesh"]
        mc = [b for b in self.colliders if b.shape == "mesh"]
        if ma and mc:
            world_c = [(c, c.world_triangles()) for c in mc]
            for a in ma:
                ta = a.world_triangles()
                for c, tc in world_c:
                    pts = compute_intersection_points(ta, tc)
                    if len(pts):
                        center = tuple(pts.mean(axis=0))
                        reports.append(CollisionReport(a, c, center, center, pts))

        touching = {}
        for r in reports:
            for body in (r.actor, r.collider):
                if body.collision_report is None:
                    body.collision_report = []
                body.collision_report.append(r)
            touching[(r.actor, r.collider)] = r
        self._fire_events(touching)
        self.reportedCollisions = reports
        return reports

    def _fire_events(self, touching):
        try:
            from direct.showbase.MessengerGlobal import messenger
        except ImportError:  # pragma: no cover
            messenger = None
        if messenger is not None:
            for key, r in touching.items():
                if key not in self._touching:
                    messenger.send("collision-enter", [r])
            for key, r in self._touching.items():
                if key not in touching:
                    messenger.send("collision-exit", [r])
        self._touching = touching

    def start(self, frame_rate=None, threaded=False):
        """
        Update automatically: every frame (or at ``frame_rate`` Hz) from a
        Panda task.  ``threaded=True`` keeps the old background-thread mode for
        use without ShowBase (do not touch the scene graph from it).
        """
        base = getattr(builtins, "base", None)
        if base is not None and not threaded:
            def step(task):
                self.update()
                return task.again if frame_rate else task.cont

            self.stop()
            if frame_rate:
                self._task = base.taskMgr.do_method_later(1.0 / frame_rate, step, "collision-world")
            else:
                self._task = base.taskMgr.add(step, "collision-world", sort=30)
            return self._task

        self._stop = False
        interval = 1.0 / (frame_rate or 60)

        def run():
            while not self._stop:
                self.update()
                time.sleep(interval)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return self._thread

    def stop(self):
        self._stop = True
        if self._task is not None:
            base = getattr(builtins, "base", None)
            if base is not None:
                base.taskMgr.remove(self._task)
            self._task = None

    def execute(self, frame_rate=60):
        self.start(frame_rate, threaded=True).join()


Mgr = CollisionWorld()
