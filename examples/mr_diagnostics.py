"""
Find which mixed-reality feature a runtime can't handle, one step at a time.

    python examples/mr_diagnostics.py

Put the headset on and look around normally. Each stage turns on one more
feature and measures frame times. At the first stall the script switches
everything off, shuts the session down cleanly (so Link survives) and says
which step it was.  Paste the output when reporting a problem.

Stages:
  1 baseline        plain VR rendering
  2 passthrough     camera passthrough layer
  3 depth provider  XR_META_environment_depth started, nothing read
  4 depth acquire   depth images acquired each frame (no GPU access)
  5 depth copy      depth images copied into Panda's texture on the GPU
  6 occlusion       real-world occlusion pass in each eye
  7 room model      Space Setup scene query + mesh
"""

import importlib.util
import logging
import pathlib
import sys
import time

_root = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "panda3d_vr", _root / "__init__.py", submodule_search_locations=[str(_root)])
panda3d_vr = importlib.util.module_from_spec(_spec)
sys.modules["panda3d_vr"] = panda3d_vr
_spec.loader.exec_module(panda3d_vr)

from panda3d_vr import BaseVrApp  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

STAGE_SECONDS = 8.0
ABORT_FRAME = 1.0  # a single frame this slow ends the run immediately


class Diagnostics(BaseVrApp):
    def __init__(self):
        super().__init__(show_stats=True, msaa=2)
        self.env = self.vr.environment_depth
        self.stages = [
            ("baseline", lambda: None),
            ("passthrough", lambda: self.vr.set_passthrough(True)),
            ("depth provider", self.start_depth),
            ("depth acquire", lambda: setattr(self.env, "acquire_enabled", True)),
            ("depth copy", lambda: setattr(self.env, "copy_enabled", True)),
            ("occlusion", lambda: self.env.set_occlusion(True)),
            ("room model", self.vr.load_scene),
        ]
        self.stage = -1
        self.results = []
        self.failed = None
        self.accept("vr-feature-disabled", self.on_disabled)
        self.taskMgr.add(self.tick, "diagnostics", sort=5)

    def start_depth(self):
        self.env.acquire_enabled = False
        self.env.copy_enabled = False
        self.vr.enable_environment_depth()

    def on_disabled(self, name, reason):
        self.failed = self.failed or "%s was switched off by the watchdog: %s" % (name, reason)

    def tick(self, task):
        vr = self.vr
        now = time.perf_counter()
        if not vr.focused:
            if self.stage < 0:
                return task.cont  # waiting for the headset
            self.failed = self.failed or "session lost focus / headset disconnected (%s)" % vr.status
        if self.stage < 0:
            self.next_stage(now)
            return task.cont
        dt = now - self.last
        self.last = now
        self.frames += 1
        self.worst = max(self.worst, dt)
        if dt > ABORT_FRAME and not self.failed:
            self.failed = "%.1f s frame during '%s'" % (dt, self.stages[self.stage][0])
        if self.failed:
            self.finish()
            return task.done
        if now - self.started >= STAGE_SECONDS:
            self.record(now)
            if self.stage + 1 >= len(self.stages):
                self.finish()
                return task.done
            self.next_stage(now)
        return task.cont

    def next_stage(self, now):
        self.stage += 1
        name, action = self.stages[self.stage]
        print("\n=== stage %d/%d: %s" % (self.stage + 1, len(self.stages), name), flush=True)
        try:
            action()
        except Exception as e:
            self.failed = "starting '%s' raised %s" % (name, e)
        self.started = self.last = now
        self.frames = 0
        self.worst = 0.0

    def record(self, now):
        name = self.stages[self.stage][0]
        fps = self.frames / max(now - self.started, 1e-6)
        env = self.env
        extra = ""
        if env is not None and env.provider is not None:
            extra = "  depth frames %d  acquire %.2f ms" % (env.frame, env.acquire_ms)
        line = "%-15s %6.1f fps   worst frame %6.1f ms   xrWaitFrame %.2f ms%s" % (
            name, fps, self.worst * 1000.0, self.vr.stats["xr_wait_ms"], extra)
        self.results.append(line)
        print(line, flush=True)

    def finish(self):
        vr = self.vr
        print("\n================ result")
        for line in self.results:
            print(line)
        if self.failed:
            print("FAILED: " + self.failed)
        else:
            scene = vr.scene
            print("All stages passed. Room anchors: %d" % (len(scene.anchors) if scene else 0))
        print("passthrough object: %s   environment depth ever valid: %s" % (
            vr.passthrough is not None, bool(self.env and self.env.frame)))
        sys.stdout.flush()
        # Clean shutdown releases depth/passthrough before the process exits.
        self.userExit()


if __name__ == "__main__":
    Diagnostics().run()
