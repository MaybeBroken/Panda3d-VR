"""
Keyboard/mouse stand-in for a headset, active whenever no OpenXR system is
available.  It drives the same head/controller nodes and fires the same
``vr-*`` events, so VR apps can be developed and tested at a desk.

    Right mouse drag   look around
    WASD               left thumbstick   (move, with Locomotion)
    Arrow keys         right thumbstick  (turn / teleport, with Locomotion)
    Left click         right trigger       F  left trigger
    E                  right squeeze       Q  left squeeze
    Space              right primary (A)   X  left primary (X)
    B                  right secondary (B) Y  left secondary (Y)
    Tab                left menu
    R / F               raise / lower hands (hold Shift)
"""

from panda3d.core import KeyboardButton, MouseButton, LVector3f

_KEY = KeyboardButton.ascii_key

# (hand index, button) -> Panda button handle
_BUTTONS = {
    (1, "trigger"): MouseButton.one(),
    (0, "trigger"): _KEY("f"),
    (1, "squeeze"): _KEY("e"),
    (0, "squeeze"): _KEY("q"),
    (1, "primary"): KeyboardButton.space(),
    (0, "primary"): _KEY("x"),
    (1, "secondary"): _KEY("b"),
    (0, "secondary"): _KEY("y"),
    (0, "menu"): KeyboardButton.tab(),
}


class DesktopSimulator:
    def __init__(self, vr, eye_height=1.65, mouse_sensitivity=0.15):
        self.vr = vr
        self.eye_height = eye_height
        self.sensitivity = mouse_sensitivity
        self.enabled = False
        self.heading = 0.0
        self.pitch = 0.0
        self.hand_offsets = [LVector3f(-0.2, 0.35, -0.3), LVector3f(0.2, 0.35, -0.3)]
        self._last_mouse = None
        self._prev_pos = [None, None]

    def enable(self):
        if self.enabled:
            return
        self.enabled = True
        vr = self.vr
        vr.head.set_pos(0, 0, self.eye_height * vr.world_scale)
        vr.head.set_hpr(self.heading, self.pitch, 0)
        for c in vr.controllers:
            c.connected = True
            c.tracked = True
            c.profile = "desktop-simulator"

    def disable(self):
        if not self.enabled:
            return
        self.enabled = False
        for c in self.vr.controllers:
            c.connected = False
            c.tracked = False
            c.profile = None
            c._reset()

    def update(self, dt):
        vr = self.vr
        base = vr.base
        mw = base.mouseWatcherNode
        down = mw.is_button_down if mw is not None else (lambda button: False)

        # Mouse look while the right button is held.
        if mw is not None and base.win is not None and mw.has_mouse() and down(MouseButton.three()):
            p = base.win.get_pointer(0)
            pos = (p.get_x(), p.get_y())
            if self._last_mouse is not None:
                self.heading -= (pos[0] - self._last_mouse[0]) * self.sensitivity
                self.pitch = max(-89.0, min(89.0, self.pitch - (pos[1] - self._last_mouse[1]) * self.sensitivity))
            self._last_mouse = pos
        else:
            self._last_mouse = None
        vr.head.set_hpr(self.heading, self.pitch, 0)

        shift = down(KeyboardButton.shift())
        if shift and down(_KEY("r")):
            for off in self.hand_offsets:
                off.z += dt * 0.5
        if shift and down(_KEY("f")):
            for off in self.hand_offsets:
                off.z -= dt * 0.5

        scale = vr.world_scale
        for i, c in enumerate(vr.controllers):
            c._begin_frame()
            before = c.grip.get_pos(vr.render)
            c.grip.set_pos(vr.head, self.hand_offsets[i] * scale)
            c.grip.set_hpr(vr.head, 0, 0, 0)
            c.aim.set_pos_hpr(c.grip, 0, 0, 0, 0, 0, 0)
            if dt > 0 and self._prev_pos[i] is not None:
                world_v = (c.grip.get_pos(vr.render) - before) / dt
                c.linear_velocity = vr.tracking_space.get_relative_vector(vr.render, world_v)
            self._prev_pos[i] = before
            for (hand, name), button in _BUTTONS.items():
                if hand == i:
                    pressed = down(button) and not (shift and name in ("trigger", "squeeze"))
                    if name in ("trigger", "squeeze"):
                        c._set_analog(name, 1.0 if pressed else 0.0)
                    else:
                        c._set_button(name, pressed)

        lx = (1.0 if down(_KEY("d")) else 0.0) - (1.0 if down(_KEY("a")) else 0.0)
        ly = (1.0 if down(_KEY("w")) else 0.0) - (1.0 if down(_KEY("s")) else 0.0)
        rx = (1.0 if down(KeyboardButton.right()) else 0.0) - (1.0 if down(KeyboardButton.left()) else 0.0)
        ry = (1.0 if down(KeyboardButton.up()) else 0.0) - (1.0 if down(KeyboardButton.down()) else 0.0)
        vr.left._set_stick(lx, ly)
        vr.right._set_stick(rx, ry)
