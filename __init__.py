"""
Panda3D-VR: OpenXR for Panda3D.

    from panda3d_vr import BaseVrApp

    class App(BaseVrApp):
        def __init__(self):
            super().__init__()
            self.accept("vr-right-trigger", lambda c: c.vibrate(0.5, 0.05))

    App().run()

Or attach to an existing ShowBase with ``VRManager(base)``.
"""

__version__ = "2.1.0"

from .controller import HANDS, Controller
from .core import EYE_MASKS, XR_AVAILABLE, BaseVrApp, VRManager
from .interaction import Interaction
from .locomotion import Locomotion
from .nodeIntersection import (
    BaseActor,
    BaseCollider,
    CollisionReport,
    CollisionWorld,
    ComplexActor,
    ComplexCollider,
    Cube,
    CubeGenerator,
    Mgr as NodeIntersection,
    Sphere,
)
from .utils import File, Math, Misc

VRApp = BaseVrApp

__all__ = [
    "BaseVrApp",
    "VRApp",
    "VRManager",
    "Controller",
    "HANDS",
    "Locomotion",
    "Interaction",
    "XR_AVAILABLE",
    "EYE_MASKS",
    "CollisionWorld",
    "NodeIntersection",
    "Sphere",
    "Cube",
    "CubeGenerator",
    "BaseActor",
    "BaseCollider",
    "ComplexActor",
    "ComplexCollider",
    "CollisionReport",
    "Math",
    "File",
    "Misc",
]
