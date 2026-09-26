"""
Per-hand controller state shared by the OpenXR backend and the desktop
simulator.

Every button fires Panda messenger events:

    vr-<hand>-<button>        on press      (e.g. "vr-right-trigger")
    vr-<hand>-<button>-up     on release    (e.g. "vr-left-primary-up")

Buttons: trigger, squeeze, primary (A/X), secondary (B/Y), menu,
thumbstick_click, thumbstick_up/down/left/right, and the capacitive
*_touch variants.  The event argument is the Controller.
"""

from panda3d.core import LVecBase2f, LVector3f
from direct.showbase.MessengerGlobal import messenger

HANDS = ("left", "right")

# Analog -> digital thresholds (press, release) for hysteresis.
ANALOG_THRESHOLDS = {"trigger": (0.55, 0.45), "squeeze": (0.55, 0.45)}
STICK_THRESHOLDS = (0.7, 0.5)


class Controller:
    """
    State for one hand.  ``grip`` and ``aim`` are NodePaths in tracking space
    that follow the controller; parent models to ``grip``, rays to ``aim``.
    """

    def __init__(self, vr, hand, parent):
        self.vr = vr
        self.hand = hand
        self.index = HANDS.index(hand)
        self.grip = parent.attach_new_node("vr-%s-grip" % hand)
        self.aim = parent.attach_new_node("vr-%s-aim" % hand)
        self.connected = False
        self.tracked = False
        self.profile = None

        self.trigger = 0.0
        self.squeeze = 0.0
        self.thumbstick = LVecBase2f(0, 0)
        self.values = {}
        self.linear_velocity = LVector3f(0, 0, 0)
        self.angular_velocity = LVector3f(0, 0, 0)

        self._buttons = {}
        self.pressed = set()
        self.released = set()
        self._prefix = "vr-%s-" % hand

    # ------------------------------------------------------------ state API

    def is_down(self, button):
        return self._buttons.get(button, False)

    def was_pressed(self, button):
        """True only on the frame the button went down."""
        return button in self.pressed

    def was_released(self, button):
        return button in self.released

    def vibrate(self, amplitude=0.5, duration=0.05, frequency=0.0):
        """Haptic pulse.  ``duration`` in seconds (-1 = shortest supported)."""
        self.vr._haptic(self.index, amplitude, duration, frequency)

    def stop_vibration(self):
        self.vr._stop_haptic(self.index)

    def get_world_velocity(self):
        return self.vr.render.get_relative_vector(self.vr.tracking_space, self.linear_velocity)

    # ------------------------------------------------------- backend hooks

    def _begin_frame(self):
        if self.pressed:
            self.pressed.clear()
        if self.released:
            self.released.clear()

    def _set_button(self, name, state):
        if self._buttons.get(name, False) == state:
            return
        self._buttons[name] = state
        if state:
            self.pressed.add(name)
            messenger.send(self._prefix + name, [self])
        else:
            self.released.add(name)
            messenger.send(self._prefix + name + "-up", [self])

    def _set_analog(self, name, value):
        self.values[name] = value
        if name == "trigger":
            self.trigger = value
        elif name == "squeeze":
            self.squeeze = value
        on, off = ANALOG_THRESHOLDS.get(name, (0.55, 0.45))
        down = self._buttons.get(name, False)
        if not down and value >= on:
            self._set_button(name, True)
        elif down and value <= off:
            self._set_button(name, False)

    def _set_stick(self, x, y):
        self.thumbstick.set(x, y)
        on, off = STICK_THRESHOLDS
        for name, v in (("thumbstick_right", x), ("thumbstick_left", -x),
                        ("thumbstick_up", y), ("thumbstick_down", -y)):
            down = self._buttons.get(name, False)
            if not down and v >= on:
                self._set_button(name, True)
            elif down and v <= off:
                self._set_button(name, False)

    def _reset(self):
        for name, state in list(self._buttons.items()):
            if state:
                self._set_button(name, False)
        self.trigger = self.squeeze = 0.0
        self.thumbstick.set(0, 0)
        self.values.clear()

    def __repr__(self):
        return "<Controller %s connected=%s profile=%s>" % (self.hand, self.connected, self.profile)
