"""
Common hand interactions.

Grabbing
    ``make_grabbable(np, radius)`` - squeeze near an object to pick it up,
    release to drop/throw it.  Events: ``vr-grab`` (np, controller) and
    ``vr-release`` (np, controller, world_velocity).

Laser pointers
    Rays from each controller's aim pose, tested against ``pointer_mask``.
    Events: ``vr-pointer-enter`` / ``vr-pointer-exit`` (np, controller) and
    ``vr-pointer-click`` (np, controller, world_point) on trigger press.
    Tag pointable NodePaths with ``np.set_python_tag("vr_pointable", True)``
    or any collision geometry under ``pointer_mask``.
"""

from panda3d.core import (
    BitMask32,
    CollisionHandlerQueue,
    CollisionNode,
    CollisionRay,
    CollisionTraverser,
)
from direct.showbase.MessengerGlobal import messenger


class Interaction:
    def __init__(self, vr, grab=True, pointers=True, pointer_mask=None, pointer_length=10.0,
                 grab_button="squeeze", click_button="trigger", haptics=True):
        self.vr = vr
        self.grab_enabled = grab
        self.pointers_enabled = pointers
        self.pointer_length = pointer_length
        self.grab_button = grab_button
        self.click_button = click_button
        self.haptics = haptics
        self._grabbables = []
        self.held = [None, None]
        self._held_parent = [None, None]
        self.hover = [None, None]
        self.hover_point = [None, None]

        mask = pointer_mask if pointer_mask is not None else CollisionNode.get_default_collide_mask()
        self._ctrav = CollisionTraverser("vr-pointers")
        self._queues = []
        self._rays = []
        for c in vr.controllers:
            node = CollisionNode("vr-pointer-%s" % c.hand)
            node.add_solid(CollisionRay(0, 0, 0, 0, 1, 0))
            node.set_from_collide_mask(mask)
            node.set_into_collide_mask(BitMask32.all_off())
            np = c.aim.attach_new_node(node)
            q = CollisionHandlerQueue()
            self._ctrav.add_collider(np, q)
            self._queues.append(q)
            self._rays.append(np)

        self._task = vr.base.taskMgr.add(self._update, "vr-interaction", sort=20)

    def destroy(self):
        self.vr.base.taskMgr.remove(self._task)
        for np in self._rays:
            np.remove_node()

    # -------------------------------------------------------------- grabbing

    def make_grabbable(self, np, radius=0.15):
        """``radius`` in metres around the object's origin."""
        np.set_python_tag("vr_grab_radius", radius)
        self._grabbables.append(np)
        return np

    def remove_grabbable(self, np):
        if np in self._grabbables:
            self._grabbables.remove(np)
        for i, h in enumerate(self.held):
            if h == np:
                self.release(i)

    def grab(self, hand, np):
        c = self.vr.controllers[hand]
        other = 1 - hand
        if self.held[other] == np:  # hand-to-hand pass keeps the original parent
            self._held_parent[hand] = self._held_parent[other]
            self.held[other] = self._held_parent[other] = None
        else:
            self._held_parent[hand] = np.get_parent()
        np.wrt_reparent_to(c.grip)
        self.held[hand] = np
        if self.haptics:
            c.vibrate(0.4, 0.03)
        messenger.send("vr-grab", [np, c])

    def release(self, hand):
        np = self.held[hand]
        if np is None:
            return
        c = self.vr.controllers[hand]
        parent = self._held_parent[hand]
        np.wrt_reparent_to(parent if parent is not None and not parent.is_empty() else self.vr.render)
        self.held[hand] = None
        self._held_parent[hand] = None
        messenger.send("vr-release", [np, c, c.get_world_velocity()])

    def _nearest(self, c):
        grip = c.grip.get_pos(self.vr.render)
        best, best_d = None, None
        scale = self.vr.world_scale
        for np in self._grabbables:
            if np.is_empty() or np.is_hidden():
                continue
            r = np.get_python_tag("vr_grab_radius") * scale
            d = (np.get_pos(self.vr.render) - grip).length()
            if d <= r and (best_d is None or d < best_d):
                best, best_d = np, d
        return best

    # ---------------------------------------------------------------- update

    def _update(self, task):
        vr = self.vr
        for i, c in enumerate(vr.controllers):
            if self.grab_enabled:
                if c.was_pressed(self.grab_button) and self.held[i] is None:
                    target = self._nearest(c)
                    if target is not None:
                        self.grab(i, target)
                elif c.was_released(self.grab_button) and self.held[i] is not None:
                    self.release(i)

        if not self.pointers_enabled:
            return task.cont
        self._ctrav.traverse(vr.render)
        for i, c in enumerate(vr.controllers):
            q = self._queues[i]
            target = point = None
            dist = self.pointer_length * vr.world_scale
            if c.connected and q.get_num_entries():
                q.sort_entries()
                for entry in q.entries:
                    into = entry.get_into_node_path()
                    if self.held[i] is not None and self.held[i].is_ancestor_of(into):
                        continue
                    p = entry.get_surface_point(vr.render)
                    d = (p - c.aim.get_pos(vr.render)).length()
                    if d > dist:
                        break
                    tagged = into.find_net_python_tag("vr_pointable")
                    target = tagged if not tagged.is_empty() else into
                    point, dist = p, d
                    break
            if target != self.hover[i]:
                if self.hover[i] is not None:
                    messenger.send("vr-pointer-exit", [self.hover[i], c])
                if target is not None:
                    messenger.send("vr-pointer-enter", [target, c])
                    if self.haptics:
                        c.vibrate(0.15, 0.01)
                self.hover[i] = target
            self.hover_point[i] = point
            if target is not None and c.was_pressed(self.click_button):
                messenger.send("vr-pointer-click", [target, c, point])
            ray = getattr(c, "ray", None)
            if ray is not None:
                ray.set_sy(dist if target is not None else 2.0 * vr.world_scale)
        return task.cont
