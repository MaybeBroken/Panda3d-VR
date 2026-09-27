"""
Check the Quest's environment depth and room model from Panda3D.

    python examples/environment_probe.py

Put the headset on. Every few seconds this prints what the depth sensor sees
(distance straight ahead, how much of the image has data) and what Space Setup
captured, and shows it in the headset: passthrough, the scanned room as a
wireframe, and a marker where the depth map says the surface in front of you is.

Over Link, enable "Passthrough over Meta Quest Link" and "Spatial data over
Meta Quest Link" in the Meta Quest Link app (Settings > Beta / Developer).
"""

import importlib.util
import pathlib
import sys

_root = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "panda3d_vr", _root / "__init__.py", submodule_search_locations=[str(_root)])
panda3d_vr = importlib.util.module_from_spec(_spec)
sys.modules["panda3d_vr"] = panda3d_vr
_spec.loader.exec_module(panda3d_vr)

import numpy as np  # noqa: E402

from panda3d_vr import BaseVrApp  # noqa: E402
from panda3d_vr.geometry import uv_sphere  # noqa: E402


class Probe(BaseVrApp):
    def __init__(self):
        super().__init__(passthrough=True, environment_depth=True, scene=True, show_stats=True)
        self.marker = self.render.attach_new_node(uv_sphere(0.03, 8, 12, (1, 0.3, 0.2, 1)))
        self.marker.set_light_off(1)
        self.accept("vr-scene-loaded", self.on_scene)
        self.accept("vr-right-trigger", lambda c: self.vr.environment_depth.set_occlusion(
            not self.vr.environment_depth.occlusion))
        self.taskMgr.do_method_later(3, self.report, "report")

    def on_scene(self, scene):
        scene.show()
        print("ROOM: %d anchors" % len(scene.anchors))
        for a in scene.anchors.values():
            print("   %-14s plane=%s volume=%s mesh=%s" % (
                a.label, a.plane and tuple(round(v, 2) for v in a.plane),
                a.volume and tuple(round(v, 2) for v in a.volume), a.mesh is not None))

    def report(self, task):
        env = self.vr.environment_depth
        if env is None or not env.valid:
            print("depth: not available yet (status=%s, focused=%s)" % (self.vr.status, self.vr.focused))
            return task.again
        metres = env.read_depth()
        left = metres[0]
        h, w = left.shape
        centre = float(np.nanmedian(np.where(np.isfinite(left[h // 2 - 2:h // 2 + 2, w // 2 - 2:w // 2 + 2]),
                                             left[h // 2 - 2:h // 2 + 2, w // 2 - 2:w // 2 + 2], np.nan)))
        print("depth: %dx%d  valid %.0f%%  straight ahead %.2f m  near %.2f far %s  occlusion %s" % (
            w, h, 100 * np.isfinite(left).mean(), centre, env.near, env.far, env.occlusion))
        def band(rows):
            vals = rows[np.isfinite(rows)]
            return float(np.median(vals)) if vals.size else float("nan")
        # Looking down at the floor, the bottom of the view is nearer. If "top"
        # reads nearer instead, the image is stored top-down: set
        # vr.environment_depth.flip_y = True.
        first, last = band(left[: h // 5]), band(left[-h // 5:])
        print("       rows 0.. (assumed bottom) %.2f m   last rows (assumed top) %.2f m" % (first, last))
        if np.isfinite(centre):
            self.marker.set_pos(env.world_point(0, 0.5, 0.5, centre))
        return task.again


if __name__ == "__main__":
    Probe().run()
