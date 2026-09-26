"""
Standard VR locomotion on the thumbsticks.

    Move hand stick     smooth movement (head- or hand-relative)
    Turn hand stick X   snap turn (or smooth turn)
    Turn hand stick Y   push forward to aim a teleport arc, release to jump

All movement rotates the rig around the *head*, so turning never swings the
player sideways.  Fires ``vr-teleport`` with the destination point.
"""

import math

from panda3d.core import (
    BitMask32,
    ClockObject,
    CollisionHandlerQueue,
    CollisionNode,
    CollisionSegment,
    CollisionTraverser,
    LineSegs,
    LPoint3f,
    LVector3f,
)
from direct.showbase.MessengerGlobal import messenger

from .geometry import uv_sphere


class Locomotion:
    def __init__(
        self,
        vr,
        move_speed=2.0,
        move_hand="left",
        turn_hand="right",
        direction="head",
        turn="snap",
        snap_angle=30.0,
        turn_speed=120.0,
        deadzone=0.2,
        fly=False,
        teleport=True,
        teleport_mask=None,
        teleport_speed=8.0,
        gravity=9.81,
        max_teleport_slope=40.0,
    ):
        """Speeds are in metres (per second) and scaled by ``vr.world_scale``."""
        self.vr = vr
        self.move_speed = move_speed
        self.move_hand = vr.left if move_hand == "left" else vr.right
        self.turn_hand = vr.right if turn_hand == "right" else vr.left
        self.direction = direction
        self.turn = turn
        self.snap_angle = snap_angle
        self.turn_speed = turn_speed
        self.deadzone = deadzone
        self.fly = fly
        self.enabled = True
        self.teleport_enabled = teleport
        self.teleport_speed = teleport_speed
        self.gravity = gravity
        self.max_teleport_slope = max_teleport_slope
        self._snap_ready = True
        self._aiming = False
        self._target = None

        self._arc_np = vr.render.attach_new_node("vr-teleport-arc")
        self._arc_np.set_light_off(1)
        self._arc_np.set_bin("fixed", 0)
        self._arc_np.hide()
        self._marker = vr.render.attach_new_node(uv_sphere(0.15, 8, 12, (0.3, 0.8, 1, 0.6), "vr-teleport-marker"))
        self._marker.set_scale(1, 1, 0.1)
        self._marker.set_transparency(True)
        self._marker.set_light_off(1)
        self._marker.hide()

        mask = teleport_mask if teleport_mask is not None else CollisionNode.get_default_collide_mask()
        self._arc_segments = 24
        self._ctrav = CollisionTraverser("vr-teleport")
        self._cqueue = CollisionHandlerQueue()
        cnode = CollisionNode("vr-teleport-ray")
        cnode.set_from_collide_mask(mask)
        cnode.set_into_collide_mask(BitMask32.all_off())
        self._segs = [CollisionSegment() for _ in range(self._arc_segments)]
        for s in self._segs:
            cnode.add_solid(s)
        self._seg_index = {id(s): i for i, s in enumerate(self._segs)}
        self._cnode_np = vr.render.attach_new_node(cnode)
        self._ctrav.add_collider(self._cnode_np, self._cqueue)

        self._task = vr.base.taskMgr.add(self._update, "vr-locomotion", sort=10)

    def destroy(self):
        self.vr.base.taskMgr.remove(self._task)
        for np in (self._arc_np, self._marker, self._cnode_np):
            np.remove_node()

    # ---------------------------------------------------------------- helpers

    def rotate_around_head(self, degrees):
        vr = self.vr
        before = vr.head.get_pos(vr.render)
        vr.rig.set_h(vr.rig.get_h() + degrees)
        after = vr.head.get_pos(vr.render)
        vr.rig.set_pos(vr.rig.get_pos() + (before - after))

    def teleport_to(self, point):
        """Move the rig so the player's feet (head projected to floor) land on ``point``."""
        vr = self.vr
        head = vr.head.get_pos(vr.render)
        rig = vr.rig.get_pos()
        vr.rig.set_pos(rig.x + point.x - head.x, rig.y + point.y - head.y, point.z)
        messenger.send("vr-teleport", [LPoint3f(point)])

    def _stick(self, ctrl):
        x, y = ctrl.thumbstick
        if x * x + y * y < self.deadzone * self.deadzone:
            return 0.0, 0.0
        return x, y

    # ----------------------------------------------------------------- update

    def _update(self, task):
        if not self.enabled:
            return task.cont
        vr = self.vr
        dt = min(ClockObject.get_global_clock().get_dt(), 0.1)
        scale = vr.world_scale

        # Smooth movement.
        x, y = self._stick(self.move_hand)
        if x or y:
            ref = vr.head if self.direction == "head" else self.move_hand.aim
            fwd = vr.render.get_relative_vector(ref, LVector3f(0, 1, 0))
            right = vr.render.get_relative_vector(ref, LVector3f(1, 0, 0))
            if not self.fly:
                fwd.z = right.z = 0
            fwd.normalize()
            right.normalize()
            vr.rig.set_pos(vr.rig.get_pos() + (fwd * y + right * x) * self.move_speed * scale * dt)

        # Turning.
        tx, ty = self._stick(self.turn_hand)
        if self.turn == "snap":
            if abs(tx) > 0.7 and self._snap_ready and not self._aiming:
                self._snap_ready = False
                self.rotate_around_head(-math.copysign(self.snap_angle, tx))
            elif abs(tx) < 0.3:
                self._snap_ready = True
        elif tx and not self._aiming:
            self.rotate_around_head(-tx * self.turn_speed * dt)

        # Teleport.
        if self.teleport_enabled:
            if ty > 0.7 and abs(tx) < 0.5:
                self._aiming = True
                self._update_arc()
            elif self._aiming and ty < 0.3:
                self._aiming = False
                self._arc_np.hide()
                self._marker.hide()
                if self._target is not None:
                    self.teleport_to(self._target)
                    self._target = None
        return task.cont

    def _update_arc(self):
        vr = self.vr
        scale = vr.world_scale
        aim = self.turn_hand.aim
        origin = aim.get_pos(vr.render)
        velocity = vr.render.get_relative_vector(aim, LVector3f(0, 1, 0))
        velocity.normalize()
        velocity *= self.teleport_speed * scale
        g = self.gravity * scale
        floor = vr.rig.get_z(vr.render)
        step = 0.06
        pts = []
        for i in range(self._arc_segments + 1):
            t = i * step
            pts.append(LPoint3f(origin.x + velocity.x * t, origin.y + velocity.y * t,
                                origin.z + velocity.z * t - 0.5 * g * t * t))
        for i, seg in enumerate(self._segs):
            seg.set_point_a(pts[i])
            seg.set_point_b(pts[i + 1])

        self._cqueue.clear_entries()
        self._ctrav.traverse(vr.render)
        hit = None
        best = len(self._segs)
        max_cos = math.cos(math.radians(self.max_teleport_slope))
        for entry in self._cqueue.entries:
            idx = self._seg_index.get(id(entry.get_from()), best)
            if idx < best:
                n = entry.get_surface_normal(vr.render)
                if n.z >= max_cos:
                    best = idx
                    hit = entry.get_surface_point(vr.render)
        end = len(pts)
        if hit is None:
            # Fall back to the rig's floor plane.
            for i in range(1, len(pts)):
                if pts[i].z <= floor < pts[i - 1].z:
                    a, b = pts[i - 1], pts[i]
                    f = (a.z - floor) / (a.z - b.z)
                    hit = a + (b - a) * f
                    end = i
                    break
        else:
            end = best + 1
        self._target = hit

        ls = LineSegs("arc")
        ls.set_thickness(3)
        ls.set_color((0.3, 0.8, 1, 1) if hit is not None else (1, 0.3, 0.3, 1))
        ls.move_to(pts[0])
        for p in pts[1:end]:
            ls.draw_to(p)
        if hit is not None:
            ls.draw_to(hit)
        self._arc_np.node().remove_all_children()
        self._arc_np.attach_new_node(ls.create())
        self._arc_np.show()
        if hit is not None:
            self._marker.set_pos(hit)
            self._marker.set_scale(scale, scale, 0.1 * scale)
            self._marker.show()
        else:
            self._marker.hide()
