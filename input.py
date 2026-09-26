"""
OpenXR action system: one action set covering every common controller,
polled each frame into ``controller.Controller`` objects, plus haptics and
eye gaze.
"""

import logging
from ctypes import POINTER, byref, c_void_p, cast, pointer

from direct.showbase.MessengerGlobal import messenger

import xr
from xr import raw_functions as raw

from .controller import HANDS
from .runtime import check
from .xrmath import apply_pose, vector_to_panda

log = logging.getLogger("panda3d_vr")

BOOL, FLOAT, VEC2, POSE, HAPTIC = "bool", "float", "vec2", "pose", "haptic"
_XR_TYPES = {
    BOOL: xr.ActionType.BOOLEAN_INPUT,
    FLOAT: xr.ActionType.FLOAT_INPUT,
    VEC2: xr.ActionType.VECTOR2F_INPUT,
    POSE: xr.ActionType.POSE_INPUT,
    HAPTIC: xr.ActionType.VIBRATION_OUTPUT,
}

# action name -> kind
DEFAULT_ACTIONS = {
    "trigger": FLOAT,
    "trigger_touch": BOOL,
    "squeeze": FLOAT,
    "thumbstick": VEC2,
    "thumbstick_click": BOOL,
    "thumbstick_touch": BOOL,
    "primary": BOOL,
    "primary_touch": BOOL,
    "secondary": BOOL,
    "secondary_touch": BOOL,
    "menu": BOOL,
    "thumbrest_touch": BOOL,
    "grip_pose": POSE,
    "aim_pose": POSE,
    "haptic": HAPTIC,
}

_POSES_AND_HAPTICS = {
    "grip_pose": "input/grip/pose",
    "aim_pose": "input/aim/pose",
    "haptic": "output/haptic",
}

# interaction profile -> {action: path relative to /user/hand/<hand>/ or {hand: path}}
PROFILES = {
    "/interaction_profiles/oculus/touch_controller": {
        "trigger": "input/trigger/value",
        "trigger_touch": "input/trigger/touch",
        "squeeze": "input/squeeze/value",
        "thumbstick": "input/thumbstick",
        "thumbstick_click": "input/thumbstick/click",
        "thumbstick_touch": "input/thumbstick/touch",
        "primary": {"left": "input/x/click", "right": "input/a/click"},
        "primary_touch": {"left": "input/x/touch", "right": "input/a/touch"},
        "secondary": {"left": "input/y/click", "right": "input/b/click"},
        "secondary_touch": {"left": "input/y/touch", "right": "input/b/touch"},
        "menu": {"left": "input/menu/click"},
        "thumbrest_touch": "input/thumbrest/touch",
        **_POSES_AND_HAPTICS,
    },
    "/interaction_profiles/valve/index_controller": {
        "trigger": "input/trigger/value",
        "trigger_touch": "input/trigger/touch",
        "squeeze": "input/squeeze/value",
        "thumbstick": "input/thumbstick",
        "thumbstick_click": "input/thumbstick/click",
        "thumbstick_touch": "input/thumbstick/touch",
        "primary": "input/a/click",
        "primary_touch": "input/a/touch",
        "secondary": "input/b/click",
        "secondary_touch": "input/b/touch",
        **_POSES_AND_HAPTICS,
    },
    "/interaction_profiles/htc/vive_controller": {
        "trigger": "input/trigger/value",
        "squeeze": "input/squeeze/click",
        "thumbstick": "input/trackpad",
        "thumbstick_click": "input/trackpad/click",
        "thumbstick_touch": "input/trackpad/touch",
        "menu": "input/menu/click",
        **_POSES_AND_HAPTICS,
    },
    "/interaction_profiles/microsoft/motion_controller": {
        "trigger": "input/trigger/value",
        "squeeze": "input/squeeze/click",
        "thumbstick": "input/thumbstick",
        "thumbstick_click": "input/thumbstick/click",
        "menu": "input/menu/click",
        **_POSES_AND_HAPTICS,
    },
    "/interaction_profiles/khr/simple_controller": {
        "trigger": "input/select/click",
        "menu": "input/menu/click",
        **_POSES_AND_HAPTICS,
    },
}

# Only suggested when XR_EXT_hand_interaction is enabled: lets tracked hands
# drive the same trigger/squeeze/pose actions as controllers.
HAND_INTERACTION_PROFILE = (
    "/interaction_profiles/ext/hand_interaction_ext",
    {
        "trigger": "input/pinch_ext/value",
        "squeeze": "input/grasp_ext/value",
        "grip_pose": "input/grip/pose",
        "aim_pose": "input/aim/pose",
    },
)

EYE_GAZE_PROFILE = "/interaction_profiles/ext/eye_gaze_interaction"
EYE_GAZE_PATH = "/user/eyes_ext/input/gaze_ext/pose"

class _Binding:
    __slots__ = ("name", "kind", "action", "get_info", "hand")


class XrInput:
    """
    Creates the action set, suggests bindings for every known controller, and
    polls it each frame with preallocated structs.

    Custom actions can be added with ``add_action`` any time before the XR
    session starts (OpenXR freezes action sets once they are attached).
    """

    def __init__(self, runtime, controllers, gaze_node=None, custom_actions=None):
        self.rt = runtime
        self.controllers = controllers
        self.gaze_node = gaze_node
        self.gaze_valid = False
        self.custom_actions = dict(custom_actions or {})
        self.action_set = None
        self.actions = {}
        self._bindings = []
        self._spaces = []
        self._attached = False
        self.accepted_profiles = []
        self.rejected_profiles = {}

    # --------------------------------------------------------------- setup

    def create(self):
        rt = self.rt
        inst = rt.instance
        self.hand_paths = [rt.path("/user/hand/left"), rt.path("/user/hand/right")]
        sub = (xr.Path * 2)(*self.hand_paths)
        self.action_set = xr.create_action_set(
            inst, xr.ActionSetCreateInfo(
                action_set_name="panda3d_vr", localized_action_set_name="Panda3D VR", priority=0))

        all_actions = dict(DEFAULT_ACTIONS)
        for name, spec in self.custom_actions.items():
            all_actions[name] = spec["kind"]
        for name, kind in all_actions.items():
            self.actions[name] = xr.create_action(
                self.action_set,
                xr.ActionCreateInfo(
                    action_name=name,
                    action_type=_XR_TYPES[kind],
                    count_subaction_paths=2,
                    subaction_paths=sub,
                    localized_action_name=name.replace("_", " ").title(),
                ),
            )

        profiles = {p: dict(b) for p, b in PROFILES.items()}
        if rt.has("hand_interaction"):
            profiles[HAND_INTERACTION_PROFILE[0]] = dict(HAND_INTERACTION_PROFILE[1])
        for name, spec in self.custom_actions.items():
            for profile, path in spec.get("bindings", {}).items():
                profiles.setdefault(profile, {})[name] = path
        for profile, table in profiles.items():
            self._suggest(profile, table)

        self.gaze_action = None
        if rt.has("eye_gaze"):
            self.gaze_action = xr.create_action(
                self.action_set,
                xr.ActionCreateInfo(action_name="eye_gaze", action_type=xr.ActionType.POSE_INPUT,
                                    localized_action_name="Eye Gaze"))
            try:
                b = xr.ActionSuggestedBinding(self.gaze_action, rt.path(EYE_GAZE_PATH))
                xr.suggest_interaction_profile_bindings(
                    inst, xr.InteractionProfileSuggestedBinding(
                        interaction_profile=rt.path(EYE_GAZE_PROFILE),
                        count_suggested_bindings=1,
                        suggested_bindings=(xr.ActionSuggestedBinding * 1)(b)))
            except Exception as e:
                log.info("Eye gaze unavailable: %s", e)
                self.gaze_action = None

    def _suggest(self, profile, table):
        rt = self.rt
        bindings = []
        for name, rel in table.items():
            if name not in self.actions:
                continue
            per_hand = rel if isinstance(rel, dict) else {h: rel for h in HANDS}
            for hand, path in per_hand.items():
                full = path if path.startswith("/") else "/user/hand/%s/%s" % (hand, path)
                bindings.append(xr.ActionSuggestedBinding(self.actions[name], rt.path(full)))
        try:
            xr.suggest_interaction_profile_bindings(
                rt.instance, xr.InteractionProfileSuggestedBinding(
                    interaction_profile=rt.path(profile),
                    count_suggested_bindings=len(bindings),
                    suggested_bindings=(xr.ActionSuggestedBinding * len(bindings))(*bindings)))
            self.accepted_profiles.append(profile)
        except Exception as e:
            self.rejected_profiles[profile] = str(e)
            log.debug("Bindings for %s rejected: %s", profile, e)

    def attach(self):
        """Create action spaces and attach; needs a session."""
        rt = self.rt
        self._spaces = []
        for hand_i, ctrl in enumerate(self.controllers):
            for name, node in (("grip_pose", ctrl.grip), ("aim_pose", ctrl.aim)):
                space = xr.create_action_space(
                    rt.session, xr.ActionSpaceCreateInfo(
                        action=self.actions[name], subaction_path=self.hand_paths[hand_i]))
                vel = xr.SpaceVelocity()
                loc = xr.SpaceLocation(next=cast(pointer(vel), c_void_p))
                self._spaces.append((space, node, loc, vel, ctrl, name == "grip_pose"))
        self._gaze_space = None
        if self.gaze_action is not None:
            self._gaze_space = xr.create_action_space(
                rt.session, xr.ActionSpaceCreateInfo(action=self.gaze_action))
            self._gaze_loc = xr.SpaceLocation()

        xr.attach_session_action_sets(
            rt.session, xr.SessionActionSetsAttachInfo(
                count_action_sets=1, action_sets=pointer(self.action_set)))
        self._attached = True

        self._active_set = xr.ActiveActionSet(self.action_set, xr.NULL_PATH)
        self._sync_info = xr.ActionsSyncInfo(
            count_active_action_sets=1, active_action_sets=pointer(self._active_set))

        self._states = {
            BOOL: xr.ActionStateBoolean(),
            FLOAT: xr.ActionStateFloat(),
            VEC2: xr.ActionStateVector2f(),
            POSE: xr.ActionStatePose(),
        }
        self._bindings = []
        for name, action in self.actions.items():
            kind = DEFAULT_ACTIONS.get(name) or self.custom_actions[name]["kind"]
            if kind == HAPTIC:
                continue
            for hand_i in range(2):
                b = _Binding()
                b.name = name
                b.kind = kind
                b.action = action
                b.hand = hand_i
                b.get_info = xr.ActionStateGetInfo(action=action, subaction_path=self.hand_paths[hand_i])
                self._bindings.append(b)

        self._haptic_info = [
            xr.HapticActionInfo(action=self.actions["haptic"], subaction_path=p) for p in self.hand_paths
        ]
        self._vibration = xr.HapticVibration()
        self._vibration_ptr = cast(pointer(self._vibration), POINTER(xr.HapticBaseHeader))
        self.refresh_profiles()

    def destroy(self):
        for entry in self._spaces:
            try:
                xr.destroy_space(entry[0])
            except Exception:
                pass
        self._spaces = []
        if getattr(self, "_gaze_space", None) is not None:
            try:
                xr.destroy_space(self._gaze_space)
            except Exception:
                pass
            self._gaze_space = None
        if self.action_set is not None:
            try:
                xr.destroy_action_set(self.action_set)
            except Exception:
                pass
        self.action_set = None
        self.actions = {}
        self._attached = False
        for c in self.controllers:
            c.connected = False
            c._reset()

    # ------------------------------------------------------------- polling

    def refresh_profiles(self):
        rt = self.rt
        for i, ctrl in enumerate(self.controllers):
            try:
                state = xr.get_current_interaction_profile(rt.session, self.hand_paths[i])
                p = state.interaction_profile
                ctrl.profile = rt.path_str(p) if p != xr.NULL_PATH else None
            except Exception:
                ctrl.profile = None

    def update(self, time, scale):
        rt = self.rt
        session = rt.session
        if not rt.focused:
            for c in self.controllers:
                c._begin_frame()
                if c.connected:
                    c.connected = False
                    c._reset()
            return
        check(raw.xrSyncActions(session, byref(self._sync_info)), "xrSyncActions")

        ctrls = self.controllers
        for c in ctrls:
            c._begin_frame()

        states = self._states
        s_bool, s_float, s_vec, s_pose = states[BOOL], states[FLOAT], states[VEC2], states[POSE]
        for b in self._bindings:
            ctrl = ctrls[b.hand]
            kind = b.kind
            if kind == FLOAT:
                raw.xrGetActionStateFloat(session, byref(b.get_info), byref(s_float))
                ctrl._set_analog(b.name, s_float.current_state if s_float.is_active else 0.0)
            elif kind == BOOL:
                raw.xrGetActionStateBoolean(session, byref(b.get_info), byref(s_bool))
                ctrl._set_button(b.name, bool(s_bool.current_state and s_bool.is_active))
            elif kind == VEC2:
                raw.xrGetActionStateVector2f(session, byref(b.get_info), byref(s_vec))
                if s_vec.is_active:
                    v = s_vec.current_state
                    if b.name == "thumbstick":
                        ctrl._set_stick(v.x, v.y)
                    else:
                        ctrl.values[b.name] = (v.x, v.y)
                elif b.name == "thumbstick":
                    ctrl._set_stick(0.0, 0.0)
            else:  # POSE
                raw.xrGetActionStatePose(session, byref(b.get_info), byref(s_pose))
                if b.name == "grip_pose":
                    active = bool(s_pose.is_active)
                    if active != ctrl.connected:
                        ctrl.connected = active
                        if not active:
                            ctrl._reset()
                        messenger.send("vr-controller-" + ("connected" if active else "disconnected"),
                                       [ctrl])

        valid_bits = xr.SPACE_LOCATION_POSITION_VALID_BIT | xr.SPACE_LOCATION_ORIENTATION_VALID_BIT
        lin_valid = xr.SPACE_VELOCITY_LINEAR_VALID_BIT
        ang_valid = xr.SPACE_VELOCITY_ANGULAR_VALID_BIT
        base = rt.space
        for space, node, loc, vel, ctrl, is_grip in self._spaces:
            raw.xrLocateSpace(space, base, time, byref(loc))
            ok = (loc.location_flags & valid_bits) == valid_bits
            if ok:
                apply_pose(node, loc.pose, scale)
            if is_grip:
                ctrl.tracked = ok
                if vel.velocity_flags & lin_valid:
                    ctrl.linear_velocity = vector_to_panda(vel.linear_velocity, scale)
                if vel.velocity_flags & ang_valid:
                    ctrl.angular_velocity = vector_to_panda(vel.angular_velocity, 1.0)

        if self._gaze_space is not None:
            gl = self._gaze_loc
            raw.xrLocateSpace(self._gaze_space, base, time, byref(gl))
            self.gaze_valid = (gl.location_flags & valid_bits) == valid_bits
            if self.gaze_valid and self.gaze_node is not None:
                apply_pose(self.gaze_node, gl.pose, scale)

    def haptic(self, hand, amplitude, duration, frequency):
        if not self._attached:
            return
        v = self._vibration
        v.amplitude = max(0.0, min(1.0, float(amplitude)))
        v.duration = -1 if duration < 0 else int(duration * 1e9)
        v.frequency = float(frequency)
        raw.xrApplyHapticFeedback(self.rt.session, byref(self._haptic_info[hand]), self._vibration_ptr)

    def stop_haptic(self, hand):
        if self._attached:
            raw.xrStopHapticFeedback(self.rt.session, byref(self._haptic_info[hand]))
