# Panda3d-VR

OpenXR support for the [Panda3D](https://www.panda3d.org/) engine. It works with Meta Quest (Link / Air Link), Valve Index, HTC Vive, Windows Mixed Reality and any other OpenXR runtime on Windows. Linux works through [WiVRn](https://github.com/WiVRn/WiVRn) or Monado (see [Linux](#linux)).

Version 2 is a full rewrite. Frames now go from Panda to the headset entirely on the GPU, and the library covers the features you would expect from a modern OpenXR integration.

## Features

### Rendering

- Zero-copy GPU pipeline. Panda renders each eye into an FBO and a single `glCopyImageSubData` moves it into the OpenXR swapchain. Nothing touches system RAM.
- Exact per-eye asymmetric projection and IPD from the runtime. The old symmetric lens and image-shift hack is gone.
- Frames are paced by `xrWaitFrame`, and poses are predicted for the actual display time.
- Recommended eye resolution, with `resolution_scale` for super- or sub-sampling and MSAA.
- Depth submission (`XR_KHR_composition_layer_depth`) for better reprojection and ASW.
- Hidden-area mesh (`XR_KHR_visibility_mask`), so the GPU skips pixels the lenses can't show.
- Quad layers: UI panels composited by the runtime at full display sharpness.
- Passthrough (`XR_FB_passthrough`) and alpha-blend / additive environment modes.
- The Quest's live depth sensing (`XR_META_environment_depth`) as a Panda texture, with automatic real-world occlusion of virtual objects.
- The Space Setup room scan (`XR_FB_scene`, `XR_META_spatial_entity_mesh`) as Panda geometry: the room mesh plus labelled walls, floor, ceiling and furniture.
- External renderers (such as mcshader's Minecraft shaderpacks) can supply the eye images themselves.
- Eye rendering is skipped automatically when the runtime says the frame won't be shown.
- Desktop mirror: `left`, `right`, `both`, `spectator` (first-person Panda camera) or `none`.

### Tracking and input

- Head, eyes, controller grip and aim poses, all as NodePaths in a clean rig hierarchy.
- Controller linear and angular velocity (for throwing).
- Full action set with bindings for Oculus/Meta Touch, Valve Index, HTC Vive, WMR, the KHR simple controller and hand-interaction profiles:
  - trigger, squeeze, thumbstick
  - A/B/X/Y and menu buttons
  - capacitive touch
  - haptics
- Every button fires Panda events (`vr-right-trigger`, `vr-left-primary-up`, ...) and also has polling APIs.
- Custom actions with your own bindings.
- Articulated hand tracking (26 joints per hand), pinch detection and a debug visualiser.
- Eye-gaze pose (`XR_EXT_eye_gaze_interaction`) on headsets that support it.
- Floor-level tracking (`local_floor`, `stage` or `local`), recentering and the guardian play-area size.

### Runtime

- Full session lifecycle: focus and visibility events, the user quitting from the headset, and user presence (headset on/off).
- Hot-plug: start without a headset, connect it later, unplug and reconnect. The app keeps running throughout.
- A keyboard/mouse desktop simulator whenever no headset is present, firing the same events so you can develop at a desk.
- Display refresh-rate query/request (72/80/90/120 Hz), CPU/GPU performance levels and an on-screen stats overlay.
- `XR_EXT_debug_utils` runtime validation messages with `debug=True`.

### Gameplay helpers

- `Locomotion`: smooth move, snap or smooth turn around the head, and a teleport arc against collision geometry or the floor.
- `Interaction`: squeeze to grab and throw, plus laser pointers with hover and click events.
- `CollisionWorld`: vectorised sphere and exact triangle-mesh intersection reporting with enter/exit events.

## Performance

These numbers come from the old and new transfer paths on the same machine, rendering the same scene. Per-frame cost includes GPU synchronisation:

| Eye resolution | v1 (RAM readback + OpenCV + re-upload) | v2 (GPU copy) | Speedup |
|---|---|---|---|
| 1000 × 1000 | 13.0 ms | 0.53 ms | ~24× |
| 2064 × 2208 (Quest 3 native) | 41.7 ms | 0.67 ms | ~62× |

At native Quest 3 resolution, v1's transfer alone capped the app at about 24 fps. In v2, almost the whole frame budget is left for your scene.

## Installation

1. Install the runtime for your headset. For Quest, install [Meta Quest Link](https://www.meta.com/help/quest/pcvr/) and set it as the active OpenXR runtime.
2. From this repository, install it as the `panda3d_vr` package. The `-e` flag keeps it pointing at the repository, so your edits apply immediately:

   ```
   pip install -e .
   ```

   Every project can then `import panda3d_vr`. The examples also run straight from the repository without installing.

### Linux

On Linux the session binds to an EGL context (`XR_MNDX_egl_enable`) rather than GLX: Monado-based runtimes only accept GLX frames through `GL_EXT_memory_object_fd`, which some drivers (Asahi on Apple Silicon) lack, while their EGL path can share images as dma-bufs. Panda's EGL desktop-GL pipe renders offscreen only, so:

- `BaseVrApp` switches Panda to `load-display p3headlessgl` and `window-type offscreen` automatically when VR is enabled. With your own `ShowBase`, put those two lines in your PRC config before creating it.
- There is no desktop window or mirror while VR is on. The keyboard/mouse simulator needs a window, so it only works with `vr=False`.

For a Quest, install WiVRn (`sudo dnf install wivrn wivrn-dashboard` on Fedora), open the `wivrn` and `mdns` services in the firewall, start the server with `systemctl --user start wivrn`, and connect from the WiVRn app on the headset.

Two platform problems you may hit:

- **Apple Silicon (Asahi):** stock WiVRn/Monado imports swapchain dma-bufs without a format modifier, which Asahi rejects (`eglCreateImageKHR failed` / `XR_ERROR_RUNTIME_FAILURE in xrCreateSwapchain`). It needs a runtime built with a patch that passes the modifier explicitly. The patch also sets `XRT_EGL_DMABUF_MODIFIER` as an override for other drivers.
- **pyopenxr on aarch64:** its wheel ships Android builds of `libopenxr_loader.so` and the API-layer libraries, which fail to load (`wrong ELF class` / `not page-aligned`). Point `xr/library/aarch64/libopenxr_loader.so` at the system loader (`/usr/lib64/libopenxr_loader.so.1`) and rebuild `libXrApiLayer_python.so` from pyopenxr's `src/generate/py_api_layer/py_api_layer.cpp`.

## Quick start

```python
from panda3d_vr import BaseVrApp, Locomotion, Interaction

class MyApp(BaseVrApp):
    def __init__(self):
        super().__init__(show_stats=True)        # any VRManager option works here
        self.loader.load_model("environment").reparent_to(self.render)

        Locomotion(self.vr)                      # sticks: move, snap turn, teleport
        Interaction(self.vr)                     # squeeze to grab, laser pointers

        self.accept("vr-right-trigger", self.fire)

    def fire(self, controller):
        controller.vibrate(amplitude=0.6, duration=0.05)
        print("aim position:", controller.aim.get_pos(self.render))

MyApp().run()
```

To add VR to an app that already has a `ShowBase`:

```python
from panda3d_vr import VRManager
vr = VRManager(base, mirror="spectator")
```

`BaseVrApp` turns off `sync-video` for you. If you use `VRManager` directly, put `sync-video false` in your PRC config, otherwise the monitor's vsync will throttle the headset.

Run the feature tour with `python examples/demo.py`. Add `--no-vr` to force the simulator.

## The rig

```
render
└── vr.rig                   move/rotate this for locomotion (alias: vr.player)
    └── vr.tracking_space    recentering offset
        ├── vr.head
        ├── vr.eye_cameras[0], vr.eye_cameras[1]
        ├── vr.left.grip / vr.left.aim      (parent models to grip, rays to aim)
        ├── vr.right.grip / vr.right.aim
        ├── vr.hands.left.joint["index_tip"] ...    (when hand tracking is active)
        └── vr.gaze
```

Units are metres by default. Use `world_scale` if your scene uses other units.

## Input

Polling:

```python
c = vr.right
c.trigger, c.squeeze, c.thumbstick          # analog values
c.is_down("primary"); c.was_pressed("trigger"); c.was_released("squeeze")
c.connected, c.tracked, c.profile           # e.g. "/interaction_profiles/oculus/touch_controller"
c.linear_velocity, c.get_world_velocity()
c.vibrate(amplitude=0.5, duration=0.1, frequency=0)
```

Events: `vr-<hand>-<button>` on press and `vr-<hand>-<button>-up` on release. The argument is the controller.

- Buttons: `trigger`, `squeeze`, `primary` (A/X), `secondary` (B/Y), `menu` and `thumbstick_click`.
- Stick directions: `thumbstick_up`, `thumbstick_down`, `thumbstick_left` and `thumbstick_right`.
- Touch variants: `trigger_touch`, `thumbstick_touch`, `primary_touch`, `secondary_touch` and `thumbrest_touch`.

Custom actions:

```python
VRManager(base, custom_actions={
    "jump": {"kind": "bool", "bindings": {
        "/interaction_profiles/oculus/touch_controller": {"right": "input/a/click"}}},
})
# -> "vr-right-jump" events and vr.right.is_down("jump")
```

## Events

| Event | Arguments | When |
|---|---|---|
| `vr-connected` / `vr-disconnected` | vr | headset found / lost (hot-plug) |
| `vr-session-created` | vr | rendering to the headset has started |
| `vr-session-state` | state name | `idle`, `ready`, `synchronized`, `visible`, `focused`, `stopping`, `exiting`... |
| `vr-focus-gained` / `vr-focus-lost` | | the system menu opened or closed |
| `vr-session-exit` | | the user quit from the headset (the app exits unless `exit_on_session_end=False`) |
| `vr-user-presence` | bool | headset put on / taken off |
| `vr-recentered` | vr | `vr.recenter()` or a runtime recenter |
| `vr-controller-connected` / `-disconnected` / `-profile` | controller | |
| `vr-<hand>-hand-tracked` / `-lost` | hand | |
| `vr-<hand>-pinch` / `-pinch-up` | hand | |
| `vr-refresh-rate-changed` | Hz | |
| `vr-eyes-ready` | vr | eye cameras and `eye_size` exist (again whenever they change); the moment to build an external renderer |
| `vr-scene-loaded` | SceneModel | the Space Setup room is in `vr.scene` |
| `vr-scene-capture-complete` | vr | the user finished Space Setup |
| `vr-feature-disabled` | name, reason | a mixed-reality feature was refused or switched off by the stall watchdog |
| `vr-teleport` | point | Locomotion |
| `vr-grab`, `vr-release` | np, controller (+ world velocity) | Interaction |
| `vr-pointer-enter` / `-exit` / `-click` | np, controller (+ point) | Interaction |
| `collision-enter` / `collision-exit` | CollisionReport | CollisionWorld |

## Options

`VRManager(base, ...)` and `BaseVrApp(...)` accept these options:

| Option | Default | |
|---|---|---|
| `world_scale` | 1.0 | Panda units per metre |
| `near`, `far` | 0.05, 1000 | clip planes in metres |
| `resolution_scale` | 1.0 | multiplier on the runtime's recommended eye size |
| `msaa` | 4 | 0 to disable |
| `tracking` | `"local_floor"` | `"stage"` or `"local"` |
| `mirror` | `"left"` | `"right"`, `"both"`, `"spectator"` or `"none"` |
| `submit_depth` | True | send depth for reprojection |
| `visibility_mask` | True | hidden-area mesh |
| `hand_tracking`, `eye_tracking` | True | used when the runtime supports them |
| `passthrough` | False | start with passthrough on (`vr.set_passthrough()` toggles it) |
| `environment_depth` | False | start depth sensing; `"occlusion"` also lets real objects hide virtual ones |
| `scene` | False | load the Space Setup room model once the app has focus |
| `blend_mode` | `"opaque"` | `"additive"` or `"alpha_blend"` for AR headsets |
| `refresh_rate` | None | request a rate in Hz |
| `fallback` | `"simulator"` | `None` for no simulator |
| `exit_on_session_end` | True | |
| `show_controllers`, `show_stats` | True, False | |
| `debug` | False | OpenXR validation messages and verbose logging |

Other calls:

- `vr.recenter()`
- `vr.set_tracking_space(kind)`
- `vr.get_play_area()`
- `vr.get_refresh_rates()` and `vr.set_refresh_rate(hz)`
- `vr.set_performance_level(cpu=..., gpu=...)`
- `vr.create_quad_layer(...)`
- `vr.set_mirror(mode)`
- `vr.enable_debug_keys()`: `r` recenters, `v` cycles the mirror mode, `F3` toggles stats, `p` toggles passthrough
- `vr.request_exit()`
- `vr.destroy()`

## Mixed reality

```python
app = BaseVrApp(passthrough=True, environment_depth="occlusion", scene=True)
```

### Environment depth

`vr.environment_depth` holds the headset's live depth map of the real world:

- `texture` is a Panda 2D texture array: layer 0 is the left eye, layer 1 the right. It's refreshed on the GPU every frame.
- `views[i]` (NodePaths) and `fov[i]` describe the depth camera that produced each layer. `near`, `far` and `params` linearise its values; `GLSL_LINEAR_DEPTH` has the matching shader helper.
- `set_occlusion(True)` writes real-world depth into each eye before the scene draws. Real furniture, walls and people then hide virtual objects behind them. This works with any shaders.
- `set_hand_removal(True)` leaves your hands out of the depth map.
- `read_depth()` returns a `(2, H, W)` array of metres on the CPU. `world_point(eye, u, v, metres)` turns an image position into a world point, for gameplay such as placing objects on real surfaces.
- `shader_inputs(np)` binds everything a custom shader needs to do its own occlusion.

You can also start it later with `vr.enable_environment_depth(occlusion=True)`.

### Room model

`vr.scene` holds the room from Space Setup:

- `mesh` is the room's triangle mesh.
- `anchors` hold each scanned entity, with its `labels` (`FLOOR`, `WALL_FACE`, `TABLE`, `COUCH`...), `plane` or `volume` extents, and a NodePath placed in the room.
- `by_label("TABLE")` finds entities by label.
- `show()` displays everything as a coloured overlay.

It loads when the app gains focus; `vr-scene-loaded` fires when it's in. `vr.request_scene_capture()` opens Space Setup in the headset.

Over Link, enable "Passthrough over Meta Quest Link" and "Spatial data over Meta Quest Link" in the Meta Quest Link app. `examples/environment_probe.py` checks both with the headset on.

### Startup safety

Over Link, depth sensing is computed from the passthrough cameras. Starting it without that feed, or before frames are flowing, can stall the runtime badly enough to drop the Link connection. To guard against this:

- Environment depth only starts once the session has been focused for a moment, and only while the passthrough feed is running. Without passthrough it logs why and stays off.
- Depth and the room model start one at a time. Each is watched for 20 seconds and switched off if:
  - frames stall (two frames over 0.5 s),
  - the frame rate drops for good (average frame time more than 1.4× what it was before the feature started), or
  - (depth only) the runtime delivers no depth image within 4 seconds. Over Link, that happens when the runtime has passthrough disabled for this PC ("Compositor passthrough is disabled (device compatibility)" in the Oculus service log): depth can't work, but the runtime's depth service still halves the frame rate.

  When that happens, `vr-feature-disabled` fires with the feature's name and the reason, and the app keeps running.

If a runtime still misbehaves, run `examples/mr_diagnostics.py` with the headset on. It enables each step in turn (passthrough, depth provider, depth acquire, GPU copy, occlusion, room model), times the frames, and stops cleanly at the first stall with a report.

## External renderers

A renderer with its own pipeline, such as a deferred shading chain or mcshader's Minecraft shaderpacks, can supply the eye images itself:

1. On `vr-eyes-ready`, render through `vr.eye_cameras[i]` (VRManager keeps their pose and exact per-eye lens current) into textures of `vr.eye_size`.
2. Call `vr.set_eye_source(i, color_texture, depth_texture=None)`.

The built-in eye rendering then idles, and your textures reach the headset through the same GPU copy (or a blit, when formats differ). `vr.get_hidden_area_camera(i)` gives the lens-mask pass to draw first, and `vr.clear_eye_sources()` switches back.

With mcshader, all of that is one argument: `mcshader.init(base, pack=..., vr=vr)`.

## Desktop simulator

When no headset is available, the simulator drives the rig and fires the same events:

| Input | Action |
|---|---|
| Right mouse drag | look around |
| WASD | left stick |
| Arrow keys | right stick |
| Left click / F | right / left trigger |
| E / Q | right / left squeeze |
| Space / X | A / X |
| B / Y | B / Y |
| Tab | menu |
| Shift+R / Shift+F | raise / lower hands |

## Migrating from v1

- `BaseVrApp` still exists, and the old constructor arguments are still accepted:
  - `wantVr` and `wantDevMode` still work.
  - `lensResolution`, `FOV` and the `auto*` flags are ignored. Resolution, FOV and eye separation now come from the runtime, and tracking is always on.
- The old attribute names still work: `player`, `vrCam`, `hand_left`, `hand_right`, `cam_left`, `cam_right` and `haptic_feedback`.
- `UpdateHeadsetTracking()` is no longer needed. Tracking updates every frame before your tasks run.
- `NodeIntersection` (`Mgr`) keeps its methods, but it now runs from a Panda task instead of a busy thread.
  - Mesh tests are now correct. v1 tetrahedralised vertex clouds.
- The `pyaudio` and OpenCV dependencies are gone. Panda's own audio plays through the headset when it is the default Windows audio device, which Link sets up.
- `Noise` (which was empty) was removed.

## Tests

```
cd tests
python -m pytest
```

The suite doesn't need a headset. A fake runtime backed by real GL textures checks what actually reaches the swapchain:

- the copy and blit paths, including external eye sources
- environment depth: the copy, CPU readback and real-world occlusion
- each eye's lens mask staying out of the other eye
- MSAA resolve
- depth values
- quad layers
- visibility masks
- skipped frames

With an OpenXR runtime installed, the suite also validates every controller binding profile against it. To include the end-to-end mcshader test, set `MCSHADER_PACK` to a shaderpack `.zip` with mcshader importable. It renders the pack per eye and checks what reaches the headset.

## Limitations

- OpenGL only: WGL on Windows, EGL on Linux (offscreen, no desktop mirror). macOS is not supported.
- Panda's single-threaded render pipeline is assumed (the default).
- Controller models are simple placeholders; parent your own models to `grip`.
- Not implemented yet: `XR_FB_render_model`, space warp (which needs motion vectors) and foveated rendering (which requires Vulkan on PC).
