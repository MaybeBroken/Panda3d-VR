"""
Articulated hand tracking (XR_EXT_hand_tracking).

Each tracked hand exposes 26 joint NodePaths in tracking space, named after
``xr.HandJointEXT`` (``palm``, ``wrist``, ``index_tip``...).  Pinch gestures
fire ``vr-<hand>-pinch`` / ``vr-<hand>-pinch-up``.
"""

import logging
from ctypes import POINTER, byref, cast

from direct.showbase.MessengerGlobal import messenger

import xr

from .runtime import check
from .xrmath import apply_pose

log = logging.getLogger("panda3d_vr")

JOINT_NAMES = tuple(name.lower() for name in xr.HandJointEXT.__members__)
JOINT_COUNT = len(JOINT_NAMES)
_THUMB_TIP = JOINT_NAMES.index("thumb_tip")
_INDEX_TIP = JOINT_NAMES.index("index_tip")

PINCH_ON = 0.018   # metres between thumb and index tips
PINCH_OFF = 0.035


class TrackedHand:
    def __init__(self, hand, parent):
        self.hand = hand
        self.root = parent.attach_new_node("vr-%s-hand" % hand)
        self.root.hide()  # shown once the runtime reports the hand as tracked
        self.joints = [self.root.attach_new_node(n) for n in JOINT_NAMES]
        self.joint = dict(zip(JOINT_NAMES, self.joints))
        self.radii = [0.0] * JOINT_COUNT
        self.active = False
        self.pinching = False
        self.pinch_strength = 0.0
        self.handle = None
        self._debug = None

    def __getitem__(self, name):
        return self.joint[name]

    def show_debug(self, sphere_model, color):
        if self._debug is None:
            self._debug = []
            for j in self.joints:
                inst = j.attach_new_node("debug")
                sphere_model.instance_to(inst)
                inst.set_color(color, 1)
                self._debug.append(inst)
        for d in self._debug:
            d.show()

    def hide_debug(self):
        for d in self._debug or ():
            d.hide()


class HandTracking:
    def __init__(self, runtime, parent):
        self.rt = runtime
        self.left = TrackedHand("left", parent)
        self.right = TrackedHand("right", parent)
        self.hands = (self.left, self.right)
        self.enabled = False

    def create(self):
        rt = self.rt
        create = rt.fn("xrCreateHandTrackerEXT", xr.PFN_xrCreateHandTrackerEXT)
        self._locate = rt.fn("xrLocateHandJointsEXT", xr.PFN_xrLocateHandJointsEXT)
        self._destroy = rt.fn("xrDestroyHandTrackerEXT", xr.PFN_xrDestroyHandTrackerEXT)
        for hand, xr_hand in zip(self.hands, (xr.HandEXT.LEFT, xr.HandEXT.RIGHT)):
            info = xr.HandTrackerCreateInfoEXT(hand=xr_hand, hand_joint_set=xr.HandJointSetEXT.DEFAULT)
            handle = xr.HandTrackerEXT()
            check(create(rt.session, byref(info), byref(handle)), "xrCreateHandTrackerEXT")
            hand.handle = handle
            hand._array = (xr.HandJointLocationEXT * JOINT_COUNT)()
            hand._locations = xr.HandJointLocationsEXT()
            hand._locations.joint_count = JOINT_COUNT
            hand._locations._joint_locations = cast(hand._array, POINTER(xr.HandJointLocationEXT))
        self._info = xr.HandJointsLocateInfoEXT()
        self.enabled = True

    def destroy(self):
        for hand in self.hands:
            if hand.handle is not None:
                try:
                    self._destroy(hand.handle)
                except Exception:
                    pass
                hand.handle = None
            hand.active = False
        self.enabled = False

    def update(self, time, scale):
        info = self._info
        info.base_space = self.rt.space
        info.time = time
        valid_bits = xr.SPACE_LOCATION_POSITION_VALID_BIT | xr.SPACE_LOCATION_ORIENTATION_VALID_BIT
        for hand in self.hands:
            locs = hand._locations
            check(self._locate(hand.handle, byref(info), byref(locs)), "xrLocateHandJointsEXT")
            active = bool(locs.is_active)
            if active != hand.active:
                hand.active = active
                if active:
                    hand.root.show()
                else:
                    hand.root.hide()
                messenger.send("vr-%s-hand-%s" % (hand.hand, "tracked" if active else "lost"), [hand])
            if not active:
                if hand.pinching:
                    hand.pinching = False
                    messenger.send("vr-%s-pinch-up" % hand.hand, [hand])
                continue
            arr = hand._array
            joints = hand.joints
            radii = hand.radii
            for i in range(JOINT_COUNT):
                loc = arr[i]
                if (loc.location_flags & valid_bits) == valid_bits:
                    apply_pose(joints[i], loc.pose, scale)
                    radii[i] = loc.radius * scale
            a = arr[_THUMB_TIP].pose.position
            b = arr[_INDEX_TIP].pose.position
            dist = ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5
            hand.pinch_strength = max(0.0, min(1.0, 1.0 - (dist - PINCH_ON) / (0.08 - PINCH_ON)))
            if not hand.pinching and dist < PINCH_ON:
                hand.pinching = True
                messenger.send("vr-%s-pinch" % hand.hand, [hand])
            elif hand.pinching and dist > PINCH_OFF:
                hand.pinching = False
                messenger.send("vr-%s-pinch-up" % hand.hand, [hand])
