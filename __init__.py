from .core import BaseVrApp, main as VrApi, xr, WANT_VR_INIT
from .utils import (
    Math,
    File,
    Misc,
    Noise,
    NodeIntersection,
    Sphere,
    Cube,
    BaseActor,
    BaseCollider,
    ComplexActor,
    ComplexCollider,
    CollisionReport,
    CubeGenerator,
)

__all__ = [
    "BaseVrApp",
    "VrApi",
    "xr",
    "Math",
    "File",
    "Misc",
    "Noise",
    "NodeIntersection",
    "Sphere",
    "BaseActor",
    "BaseCollider",
    "ComplexActor",
    "ComplexCollider",
    "CollisionReport",
    "Cube",
    "CubeGenerator",
    "WANT_VR_INIT",
]

if __name__ == "__main__":
    print("Panda3d VR Interface -- Version {VrApiVersion}")
