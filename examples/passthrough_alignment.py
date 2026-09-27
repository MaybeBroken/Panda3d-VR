"""
Check how the passthrough "holes" line up with what the app draws.

    python examples/passthrough_alignment.py

Three shapes float in front of you, fixed to your view, over passthrough:

    RED square      up and to the LEFT
    GREEN square    down and to the RIGHT
    BLUE bar        across the BOTTOM

Wherever the app draws, the camera image should be hidden. Note where the
camera image is cut out compared with each shape:
    - exactly behind it           -> aligned
    - mirrored top/bottom         -> alpha is flipped vertically
    - mirrored left/right         -> alpha is flipped horizontally
    - shifted by a fixed amount   -> offset (say which way and how far)
Press the right trigger (or A) to quit.
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

from panda3d.core import CardMaker  # noqa: E402

from panda3d_vr import BaseVrApp  # noqa: E402


class Alignment(BaseVrApp):
    def __init__(self):
        super().__init__(passthrough=True, show_controllers=False, show_stats=True)
        self.setBackgroundColor(0, 0, 0, 0)
        for name, frame, color in (
            ("red", (-0.55, -0.25, 0.20, 0.50), (1, 0, 0, 1)),
            ("green", (0.25, 0.55, -0.45, -0.15), (0, 1, 0, 1)),
            ("blue", (-0.60, 0.60, -0.62, -0.55), (0, 0.3, 1, 1)),
        ):
            cm = CardMaker(name)
            cm.set_frame(*frame)
            card = self.vr.head.attach_new_node(cm.generate())
            card.set_y(1.2)  # 1.2 m in front of the eyes, follows the head
            card.set_color(color)
            card.set_light_off(1)
            card.set_two_sided(True)
        self.accept("vr-right-trigger", lambda c: self.userExit())
        self.accept("vr-right-primary", lambda c: self.userExit())
        print(__doc__)


if __name__ == "__main__":
    Alignment().run()
