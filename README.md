# Panda3d-VR

This is an alpha API to link a Meta Quest 3 to the Panda3d game engine.

It is very much in the development stage, but it gets the job done.

I should mention there is NO documentation, so you will have to look at the code to figure out how to use it... you're welcome:D

The code is a bit messy, and I am not a professional programmer, so please be kind. I am also not a VR developer, so if you have any suggestions on how to improve the code, please let me know.

## Features

- Full 3d rendering
- Headset motion tracking
- Controller tracking
- Basic controller interface, able to detect the trigger and haptics
- Player control
- New base class to parent from instead of `ShowBase`

## Limitations

- Movement is a bit broken, and if you tilt the headset too far up (like completely vertical) some strange things happen
- Unable to do much with controller buttons
- It requires the OpenXR runtime, which in this case requires the program to be Windows-only and have at least a GTX 1080 and Meta Link software installed
- The code is still in alpha, and any currently stated things are liable to change

## Installation

Install the Meta Link software [here](https://www.meta.com/help/quest/pcvr/)

Enable Developer mode on your headset, tutorial [here](https://medium.com/sidequestvr/how-to-turn-on-developer-mode-for-the-quest-3-509244ccd386)

Clone from GitHub, and use the folder as a module in your program.

## Python example

```python
from panda3d.core import *
from api import *

class MyVrApp(BaseVrApp):
  def __init__(self):
    super().__init__(
      lensResolution=(1000, 1000),
      wantDevMode=False,
      FOV=95.5,
      autoCamPositioning=False,
      autoCamRotation=False,
      autoControllerPositioning=False,
      autoControllerRotation=False,
      launchShowBase=True,
      wantVr=True,
    )
    # Your panda3d code here

MyVrApp().run()
```

## Usage

This is just a layer over the default `ShowBase` class, with some extra utilities

Documentation to be added... in the future. Maybe. For now, the lens is `self.vrLens`, the main player is `self.player`, and there is a `self.cam_left`, `self.cam_right`, as well as a NodePath `self.vrCam`.
The rendering is done in `self.buffer_left` and `self.buffer_right`, and the live render textures are available in `self.cam_left_tex` and `self.cam_right_tex`
