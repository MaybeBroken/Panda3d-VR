"""
Panda3D-VR feature tour.

    python examples/demo.py              # headset if connected, else desktop simulator
    python examples/demo.py --no-vr      # force the desktop simulator
    python examples/demo.py --mr         # mixed reality: passthrough, real-world
                                         # occlusion and the scanned room

Try: thumbsticks to move/snap-turn, push the right stick forward to teleport,
squeeze to grab cubes, point + trigger at the spheres, A to toggle
passthrough, B to recenter, left menu to cycle the mirror mode, Y to show the
scanned room.
"""

import importlib.util
import pathlib
import random
import sys

# Load the repo folder as the "panda3d_vr" package (its name has a hyphen).
_root = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "panda3d_vr", _root / "__init__.py", submodule_search_locations=[str(_root)])
panda3d_vr = importlib.util.module_from_spec(_spec)
sys.modules["panda3d_vr"] = panda3d_vr
_spec.loader.exec_module(panda3d_vr)

from panda3d.core import (  # noqa: E402
    AmbientLight,
    CardMaker,
    CollisionNode,
    CollisionPlane,
    CollisionSphere,
    DirectionalLight,
    NodePath,
    Plane,
    TextNode,
)

from panda3d_vr import BaseVrApp, Interaction, Locomotion  # noqa: E402
from panda3d_vr.geometry import box, uv_sphere  # noqa: E402


class Demo(BaseVrApp):
    def __init__(self, vr=True, mixed_reality=False):
        mr = dict(passthrough=True, environment_depth="occlusion", scene=True) if mixed_reality else {}
        super().__init__(vr=vr, show_stats=True, debug_keys=True, mirror="left", **mr)
        self.setBackgroundColor(0.05, 0.07, 0.1)
        self._build_scene()
        if mixed_reality:
            self.floor.hide()  # the real floor shows through passthrough

        self.locomotion = Locomotion(self.vr)
        self.interaction = Interaction(self.vr)
        for cube in self.cubes:
            self.interaction.make_grabbable(cube, radius=0.15)

        self.accept("vr-right-primary", lambda c: self.vr.set_passthrough(not self.vr.want_passthrough))
        self.accept("vr-right-secondary", lambda c: self.vr.recenter())
        self.accept("vr-left-menu", lambda c: self.vr.cycle_mirror())
        self.accept("vr-right-trigger", lambda c: c.vibrate(0.3, 0.02))
        self.accept("vr-pointer-enter", lambda np, c: np.set_color_scale(1.6, 1.6, 1.6, 1))
        self.accept("vr-pointer-exit", lambda np, c: np.clear_color_scale())
        self.accept("vr-pointer-click", self._on_click)
        self.accept("vr-release", self._on_release)
        self.accept("vr-session-created", self._on_session)
        self.accept("vr-left-pinch", lambda hand: print("left pinch"))
        self.accept("vr-left-secondary", lambda c: self._toggle_room())
        self.accept("vr-scene-loaded", lambda scene: print(
            "room: %d anchors %s" % (len(scene.anchors), sorted({a.label for a in scene.anchors.values()}))))

    def _toggle_room(self):
        scene = self.vr.scene
        if scene is None:
            return
        if not scene.anchors:
            self.vr.load_scene()
        (scene.hide if not scene.root.is_hidden() else scene.show)()

    def _build_scene(self):
        render = self.render
        cm = CardMaker("floor")
        cm.set_frame(-20, 20, -20, 20)
        floor = self.floor = render.attach_new_node(cm.generate())
        floor.set_p(-90)
        floor.set_color(0.25, 0.3, 0.35, 1)
        col = render.attach_new_node(CollisionNode("floor-col"))
        col.node().add_solid(CollisionPlane(Plane((0, 0, 1), (0, 0, 0))))

        grid = render.attach_new_node("grid")
        color = (0.4, 0.5, 0.6, 1)
        for i in range(-10, 11):
            grid.attach_new_node(box((0.005, 10, 0.001), (i, 0, 0.001), color))
            grid.attach_new_node(box((10, 0.005, 0.001), (0, i, 0.001), color))
        grid.flatten_strong()  # 42 boxes -> one batch

        self.cubes = []
        for i in range(6):
            c = render.attach_new_node(box((0.08, 0.08, 0.08), (0, 0, 0),
                                           (random.random(), random.random(), random.random(), 1)))
            c.set_pos(-0.6 + i * 0.25, 0.8, 1.0)
            self.cubes.append(c)

        self.targets = []
        for i in range(5):
            s = render.attach_new_node(uv_sphere(0.25, 16, 24, (0.9, 0.5, 0.2, 1)))
            s.set_pos(-4 + i * 2, 6, 1.5)
            cn = s.attach_new_node(CollisionNode("target"))
            cn.node().add_solid(CollisionSphere(0, 0, 0, 0.25))
            s.set_python_tag("vr_pointable", True)
            self.targets.append(s)

        sun = DirectionalLight("sun")
        sun.set_color((0.9, 0.9, 0.85, 1))
        sun_np = render.attach_new_node(sun)
        sun_np.set_hpr(30, -50, 0)
        amb = AmbientLight("amb")
        amb.set_color((0.35, 0.35, 0.4, 1))
        render.set_light(sun_np)
        render.set_light(render.attach_new_node(amb))

    def _on_click(self, np, controller, point):
        np.set_color(random.random(), random.random(), random.random(), 1)
        controller.vibrate(0.8, 0.05)

    def _on_release(self, np, controller, velocity):
        print("released %s at %.2f m/s" % (np.get_name(), velocity.length()))

    def _on_session(self, vr):
        # A crisp compositor quad showing headset info, floating in front of the start pose.
        panel = vr.create_quad_layer(resolution=(1024, 256), size=(1.0, 0.25))
        panel.node.set_pos(0, 1.5, 1.9)
        text = TextNode("info")
        text.set_text("%s\n%s  |  hands: %s  |  refresh: %s Hz" % (
            vr.system_name, vr.rt.runtime_name, "yes" if vr.hands else "no", vr.get_refresh_rate()))
        text.set_align(TextNode.A_center)
        tnp = panel.root.attach_new_node(text)
        tnp.set_scale(0.3)
        tnp.set_pos(0, 0, 0.2)
        if vr.hands is not None:
            model = NodePath(uv_sphere(0.008, 6, 8))
            vr.hands.left.show_debug(model, (0.3, 0.6, 1, 1))
            vr.hands.right.show_debug(model, (1, 0.45, 0.3, 1))


if __name__ == "__main__":
    Demo(vr="--no-vr" not in sys.argv, mixed_reality="--mr" in sys.argv).run()
