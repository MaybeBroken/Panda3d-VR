WANT_VR_INIT = False
import sys

if sys.platform == "win32":
    from OpenGL import GL
    import xr
else:
    print("VR modules failed to load.")
    if "a" == "b":
        from OpenGL import GL
        import xr
import numpy as np
import cv2
from threading import Thread as _Thread

from panda3d.core import *
from panda3d.core import (
    GraphicsPipe,
    FrameBufferProperties,
    WindowProperties,
    GraphicsOutput,
    GraphicsEngine,
    PNMImage,
    Texture,
    PerspectiveLens,
    ConfigVariableString,
    LVecBase3,
    LQuaternionf,
)
from direct.showbase.ShowBase import ShowBase
from time import sleep
import ctypes
from math import asin, degrees, radians, sqrt, sin, cos, tan, sinh, cosh, tanh
import os
from ctypes import pointer, Structure, c_float, POINTER, cast, byref
import enum
import atexit
from .utils import *

if sys.platform == "win32":
    import pyaudio
else:
    print("Audio not yet supported on this platform.")
    if "a" == "b":
        import pyaudio  # type: ignore


class Side(enum.IntEnum):
    LEFT = 0
    RIGHT = 1


class InputState(Structure):
    def __init__(self):
        super().__init__()
        self.hand_scale[:] = [1, 1]
    
    if sys.platform == "win32":
        _fields_ = [
            ("action_set", xr.ActionSet),
            ("grab_action", xr.Action),
            ("pose_action", xr.Action),
            ("vibrate_action", xr.Action),
            ("quit_action", xr.Action),
            ("hand_subaction_path", xr.Path * len(Side)),
            ("hand_space", xr.Space * len(Side)),
            ("hand_scale", c_float * len(Side)),
            ("hand_active", xr.Bool32 * len(Side)),
        ]
        action_set = None
        grab_action = None
        pose_action = None
        vibrate_action = None
        quit_action = None
        hand_subaction_path = None
        hand_space = None
        hand_scale = None
        hand_active = None
        hand_triggers: dict = None


class Hand:
    def __init__(self):
        self.active = False
        self.trigger_value = 0.0
        self.haptic_strength = 1
        self.haptic_threshold = 0.9
        self.haptic_frequency = 150


class HandControl:
    def __init__(self):
        self.hand_left = Hand()
        self.hand_right = Hand()
        self.hands = [self.hand_left, self.hand_right]

    def get_hands(self):
        return self.hands


class BaseVrApp(ShowBase):

    def __init__(
        self,
        lensResolution=(1000, 1000),
        wantDevMode=False,
        FOV=95.5,
        autoCamPositioning=False,
        autoCamRotation=False,
        autoControllerPositioning=False,
        autoControllerRotation=False,
        launchShowBase=True,
        wantVr=WANT_VR_INIT,
    ):
        WANT_VR_INIT = wantVr
        # NVidia driver requires this env variable to be set to 0 to disable v-sync
        os.environ["__GL_SYNC_TO_VBLANK"] = "0"
        # MESA OpenGL drivers requires this env variable to be set to 0 to disable v-sync
        os.environ["vblank_mode"] = "0"
        ConfigVariableString("", "textures-power-2 0")
        if launchShowBase:
            super().__init__()
        # else:
        #     pipeline_path = "./render_pipeline"
        #     sys.path.insert(0, pipeline_path)

        #     from render_pipeline import RenderPipeline

        #     self.render_pipeline = RenderPipeline()
        #     self.render_pipeline.create(self)
        #     self.render_pipeline.daytime_mgr.time = "7:40"
        #     self.render_pipeline.set_effect(
        #         self.render, "src/scene-effect.yaml", {}, sort=250
        #     )
        if sys.platform == "win32" and WANT_VR_INIT:
            _Thread(target=main.start, daemon=True).start()
        else:
            print("VR not supported or disabled on this platform.")
        self.disableMouse()
        self.setBackgroundColor(0, 0, 0)
        self.lensResolution = lensResolution
        self.wantDevMode = wantDevMode
        self.FOV = FOV

        self.camList = []
        self.view_left = True  # Flag to track which buffer to display

        # Create the left camera buffer
        self.vrLens = PerspectiveLens()
        self.buffer_left = self.make_buffer()
        self.cam_left = self.makeCamera(
            self.buffer_left, scene=self.render, lens=self.vrLens
        )
        self.cam_left.setPos(-0.25, 0, 0)  # Adjusted offset to the left
        self.camList.append(self.cam_left)

        # Create the right camera buffer
        self.buffer_right = self.make_buffer()
        self.cam_right = self.makeCamera(
            self.buffer_right, scene=self.render, lens=self.vrLens
        )
        self.cam_right.setPos(0.25, 0, 0)  # Adjusted offset to the right
        self.camList.append(self.cam_right)

        self.cam_left_tex = Texture()
        self.cam_right_tex = Texture()
        self.cam_left_tex.setKeepRamImage(True)
        self.cam_right_tex.setKeepRamImage(True)
        global cam_left_tex, cam_right_tex
        cam_left_tex = self.cam_left_tex
        cam_right_tex = self.cam_right_tex
        self.buffer_left.addRenderTexture(
            self.cam_left_tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPColor
        )
        self.buffer_right.addRenderTexture(
            self.cam_right_tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPColor
        )
        self.player = self.render.attachNewNode("vrCamRoot")
        self.vrCam = self.player.attachNewNode("vrCam")
        self.cam_left.reparentTo(self.vrCam)
        self.cam_right.reparentTo(self.vrCam)
        self.camRootNode = self.player.attachNewNode("camRootNode")
        self.vrCam.reparentTo(self.camRootNode)
        self.camera.reparentTo(self.camRootNode)
        self.vrCameraPose = None

        self.vrCamPos = (0, 0, 0)
        self.vrCamHpr = (0, 0, 0)
        self.vrCamPosOffset = (0, 0, -1.4)
        self.vrControllerPosOffset = (0, 0, -1.35)
        self.vrCamHprOffset = (0, 0, 0)
        self.vrControllerHprOffset = (0, 0, 0)

        self.lastException = ""
        self.lastExceptionTime = 0

        self.handRootNode = self.player.attachNewNode("handRootNode")
        self.hand_left = self.handRootNode.attachNewNode("hand_left")
        self.hand_right = self.handRootNode.attachNewNode("hand_right")
        self.vrControllerPose = None

        self.autoCamPositioning = autoCamPositioning
        self.autoCamRotation = autoCamRotation
        self.autoControllerPositioning = autoControllerPositioning
        self.autoControllerRotation = autoControllerRotation

        if wantDevMode:
            self.doMethodLater(
                2, lambda task: self.toggle_dev_win_view(), "toggle_dev_win_view"
            )
            self.accept("v", self.toggle_dev_win_view)
            self.accept(
                "p",
                lambda: print(
                    f"Lens FOV: {self.vrLens.getFov()}\nLens Distance: {self.cam_left.getX(), self.cam_right.getX()}\nImage Offset: {main.image_offset}"
                ),
            )
            self.accept(
                "wheel_up",
                lambda: (
                    self.vrLens.setFov(self.vrLens.getFov() + 0.5),
                    print("FOV: ", self.vrLens.getFov()),
                ),
            )
            self.accept(
                "wheel_down",
                lambda: (
                    self.vrLens.setFov(self.vrLens.getFov() - 0.5),
                    print("FOV: ", self.vrLens.getFov()),
                ),
            )
            self.accept(
                "control-wheel_up",
                lambda: (
                    main.update_image_offset(0.01),
                    print("Offset: ", main.image_offset),
                ),
            )
            self.accept(
                "control-wheel_down",
                lambda: (
                    main.update_image_offset(-0.01),
                    print("Offset: ", main.image_offset),
                ),
            )
            self.accept(
                "alt-wheel_up",
                lambda: (
                    self.cam_left.setPos(self.cam_left.getX() - 0.01, 0, 0),
                    self.cam_right.setPos(self.cam_right.getX() + 0.01, 0, 0),
                    print(
                        "Lens Distance: ", (self.cam_left.getX(), self.cam_right.getX())
                    ),
                ),
            )
            self.accept(
                "alt-wheel_down",
                lambda: (
                    self.cam_left.setPos(self.cam_left.getX() + 0.01, 0, 0),
                    self.cam_right.setPos(self.cam_right.getX() - 0.01, 0, 0),
                    print(
                        "Lens Distance: ", (self.cam_left.getX(), self.cam_right.getX())
                    ),
                ),
            )
            self.accept("r", self.reset_view_orientation)

        self.vrLens.setFov(FOV)
        self.vrLens.setAspectRatio(self.lensResolution[0] / self.lensResolution[1])
        self.cam_left.setPos(-0.25, 0, 0)
        self.cam_right.setPos(0.25, 0, 0)
        self.vrCamPos = (0, 0, 0)
        self.vrCamHpr = (0, 0, 0)
        if sys.platform == "win32":
            if WANT_VR_INIT:
                self.HandState: list[Hand, Hand] = main.HandControl.get_hands()
            else:
                self.HandState: list[Hand, Hand] = HandControl().get_hands()
            self.HandControl = HandControl()
        else:
            None

    def haptic_feedback(
        self, hand: Side, amplitude: float, duration: int, frequency: int
    ):
        if sys.platform == "win32" and WANT_VR_INIT:
            main.haptic_feedback(hand, amplitude, duration, frequency)

    def UpdateHeadsetTracking(self):
        if sys.platform == "win32" and WANT_VR_INIT:
            try:
                while main.pose_quat is None:
                    GraphicsEngine.get_global_ptr().render_frame()
                    sleep(0.01)
                self.vrCameraPose = main.pose_quat
                self.vrControllerPose = main.controller
                self.camera.setPos(self.vrCam.getPos())
                self.camera.setHpr(self.vrCam.getHpr())
                self.camLens.setFov(self.vrLens.getFov())
                self.camLens.setAspectRatio(
                    self.lensResolution[0] / self.lensResolution[1]
                )
                self.HandState: list[Hand, Hand] = main.HandControl.get_hands()
                main.HandControl = self.HandControl

                if self.autoCamPositioning:
                    self.vrCam.setPos(
                        ((self.vrCameraPose.position.x + self.vrCamPosOffset[0]) * 8)
                        + self.vrCamPos[0],
                        ((self.vrCameraPose.position.z + self.vrCamPosOffset[1]) * -8)
                        + self.vrCamPos[1],
                        ((self.vrCameraPose.position.y + self.vrCamPosOffset[2]) * 8)
                        + self.vrCamPos[2],
                    )
                if self.autoCamRotation:
                    self.vrCam.setQuat(
                        LQuaternionf(
                            -self.vrCameraPose.orientation.w,
                            -self.vrCameraPose.orientation.x,
                            -self.vrCameraPose.orientation.z,
                            -self.vrCameraPose.orientation.y,
                        )
                    )
                    vrCamH = self.vrCam.getH()
                    vrCamP = self.vrCam.getP()
                    vrCamR = self.vrCam.getR()
                    self.vrCam.setR(-vrCamR * cos(radians(vrCamH)))
                    self.vrCam.setR(self.vrCam, vrCamP * cos(radians(vrCamH + 90)))
                    self.vrCam.setP(vrCamP * cos(radians(vrCamH)))
                    self.vrCam.setP(self.vrCam, vrCamR * cos(radians(vrCamH + 90)))
                if self.autoControllerPositioning:
                    try:
                        self.hand_left.setPos(
                            (
                                (self.vrControllerPose["left"].position.x)
                                + self.vrControllerPosOffset[0]
                            )
                            * 9,
                            (
                                (self.vrControllerPose["left"].position.z)
                                + self.vrControllerPosOffset[1]
                            )
                            * -9,
                            (
                                (self.vrControllerPose["left"].position.y)
                                + self.vrControllerPosOffset[2]
                            )
                            * 9,
                        )
                        self.hand_right.setPos(
                            (
                                (self.vrControllerPose["right"].position.x)
                                + self.vrControllerPosOffset[0]
                            )
                            * 9,
                            (
                                (self.vrControllerPose["right"].position.z)
                                + self.vrControllerPosOffset[1]
                            )
                            * -9,
                            (
                                (self.vrControllerPose["right"].position.y)
                                + self.vrControllerPosOffset[2]
                            )
                            * 9,
                        )
                    except:
                        pass
                if self.autoControllerRotation:
                    try:
                        self.hand_left.setQuat(
                            LQuaternionf(
                                -self.vrControllerPose["left"].orientation.w,
                                -self.vrControllerPose["left"].orientation.x,
                                -self.vrControllerPose["left"].orientation.z,
                                -self.vrControllerPose["left"].orientation.y,
                            )
                        )
                        self.hand_left.setR(-self.hand_left.getR())
                        self.hand_right.setQuat(
                            LQuaternionf(
                                -self.vrControllerPose["right"].orientation.w,
                                -self.vrControllerPose["right"].orientation.x,
                                -self.vrControllerPose["right"].orientation.z,
                                -self.vrControllerPose["right"].orientation.y,
                            )
                        )
                        self.hand_right.setR(-self.hand_right.getR())
                    except:
                        pass
            except Exception as e:
                print(str(e))
                sleep(0.1)

    def resetView(self):
        self.vrLens.setFov(self.FOV)
        self.vrLens.setAspectRatio(self.lensResolution[0] / self.lensResolution[1])
        self.cam_left.setPos(-0.25, 0, 0)
        self.cam_right.setPos(0.25, 0, 0)

    def reset_view_orientation(self):
        self.vrCamPosOffset = (
            -self.vrCameraPose.position.x,
            -self.vrCameraPose.position.y,
            -self.vrCameraPose.position.z,
        )
        self.vrCamHprOffset = (
            -self.vrCameraPose.orientation.y,
            -self.vrCameraPose.orientation.x,
            -self.vrCameraPose.orientation.z,
        )
        self.resetView()

    def make_buffer(self, lensResolution=None):
        if lensResolution is None:
            lensResolution = self.lensResolution
        winprops = WindowProperties.size(
            lensResolution[0],
            lensResolution[1],
        )
        fbprops = FrameBufferProperties()
        fbprops.setRgbColor(True)
        fbprops.setDepthBits(1)
        buffer = self.graphicsEngine.makeOutput(
            self.pipe,
            "offscreen buffer",
            -2,
            fbprops,
            winprops,
            GraphicsPipe.BFRefuseWindow,
            self.win.getGsg(),
            self.win,
        )
        return buffer

    def toggle_dev_win_view(self):
        def update_frames():
            while True:
                try:
                    frame_left = self.get_camera_image(self.cam_left_tex)
                    frame_right = self.get_camera_image(self.cam_right_tex)

                    frame_data_left = np.flipud(
                        np.array(frame_left, dtype=np.uint8)[:, :, ::-1]
                    )
                    frame_data_right = np.flipud(
                        np.array(frame_right, dtype=np.uint8)[:, :, ::-1]
                    )

                    # Create or update windows to display the frames
                    left_window_name = "Left Lens Buffer"
                    right_window_name = "Right Lens Buffer"
                    cv2.namedWindow(left_window_name, cv2.WINDOW_NORMAL)
                    cv2.namedWindow(right_window_name, cv2.WINDOW_NORMAL)
                    cv2.imshow(left_window_name, frame_data_left)
                    cv2.imshow(right_window_name, frame_data_right)

                    # Add a small delay to allow the window to refresh
                    cv2.waitKey(1)
                except Exception as e:
                    print(str(e))
                    print()
                    sleep(0.1)

        # Run the frame update in a separate thread to avoid blocking the main loop
        _Thread(target=update_frames, daemon=True).start()

    def get_camera_image(self, texture):
        while not texture.hasRamImage():
            sleep(0.01)
        image = np.array(texture.getRamImageAs("RGB"), dtype=np.uint8)
        image = image.reshape((texture.getYSize(), texture.getXSize(), 3))
        return image

    def playSound(self, sound):
        sound_data = sound.getRawData()
        sound_array = np.frombuffer(sound_data, dtype=np.int16)
        main.audio_stream.write(sound_array.tobytes())


class main:
    def start(self):
        self.image_offset = 0.1225
        self.lastException = ""
        self.lastExceptionTime = 0
        self.pose_quat = None
        self.controller = {"left": None, "right": None}
        self.context = None  # Add this line to store the context object
        self.session = None
        self.instance = None
        self.input = InputState()
        self.HandControl: HandControl = HandControl()

        while True:
            try:
                cam_left_tex
                cam_right_tex
                break
            except NameError as e:
                print(str(e))
                print("\n")
                sleep(0.1)

        frame_left = self.get_camera_image(cam_left_tex)
        frame_right = self.get_camera_image(cam_right_tex)

        frame_data_left = np.array(frame_left, np.uint8)
        frame_data_right = np.array(frame_right, np.uint8)

        # ContextObject is a high level pythonic class meant to keep simple cases simple.
        with xr.ContextObject(
            instance_create_info=xr.InstanceCreateInfo(
                enabled_extension_names=[
                    # A graphics extension is mandatory (without a headless extension)
                    xr.KHR_OPENGL_ENABLE_EXTENSION_NAME,
                ],
            ),
        ) as context:
            self.initialize_audio()
            self.session = context.session
            self.context = context
            self.instance = context.instance
            texture_id_left = GL.glGenTextures(1)
            texture_id_right = GL.glGenTextures(1)
            self.initialize_actions()

            def setup_texture(texture_id):
                GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
                GL.glTexParameteri(
                    GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR
                )
                GL.glTexParameteri(
                    GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
                )

            setup_texture(texture_id_left)
            setup_texture(texture_id_right)

            VERTEX_SHADER_SOURCE = """
            #version 330 core
            layout(location = 0) in vec3 position;
            layout(location = 1) in vec2 texCoord;
            out vec2 TexCoord;
            void main()
            {
                gl_Position = vec4(position, 1.0);
                TexCoord = texCoord;
            }
            """

            FRAGMENT_SHADER_SOURCE = """
            #version 330 core
            in vec2 TexCoord;
            out vec4 color;
            uniform sampler2D ourTexture;
            uniform vec2 offset;
            void main()
            {
                color = texture(ourTexture, TexCoord + offset);
            }
            """

            shader_program = self.create_shader_program(
                VERTEX_SHADER_SOURCE,
                FRAGMENT_SHADER_SOURCE,
            )
            GL.glUseProgram(shader_program)

            vertices = np.array(
                [
                    # positions       # texture coords
                    -1.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    -1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    -1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                dtype=np.float32,
            )

            indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

            VAO = GL.glGenVertexArrays(1)
            VBO = GL.glGenBuffers(1)
            EBO = GL.glGenBuffers(1)

            GL.glBindVertexArray(VAO)

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
            GL.glBufferData(
                GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW
            )

            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, EBO)
            GL.glBufferData(
                GL.GL_ELEMENT_ARRAY_BUFFER,
                indices.nbytes,
                indices,
                GL.GL_STATIC_DRAW,
            )

            GL.glVertexAttribPointer(
                0,
                3,
                GL.GL_FLOAT,
                GL.GL_FALSE,
                5 * vertices.itemsize,
                GL.ctypes.c_void_p(0),
            )
            GL.glEnableVertexAttribArray(0)

            GL.glVertexAttribPointer(
                1,
                2,
                GL.GL_FLOAT,
                GL.GL_FALSE,
                5 * vertices.itemsize,
                GL.ctypes.c_void_p(3 * vertices.itemsize),
            )
            GL.glEnableVertexAttribArray(1)

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glBindVertexArray(0)

            for frame_index, frame_state in enumerate(context.frame_loop()):
                view_state, views = xr.locate_views(
                    session=context.session,
                    view_locate_info=xr.ViewLocateInfo(
                        view_configuration_type=context.view_configuration_type,
                        display_time=frame_state.predicted_display_time,
                        space=context.space,
                    ),
                )
                flags = xr.ViewStateFlags(view_state.view_state_flags)
                if flags & xr.ViewStateFlags.ORIENTATION_VALID_BIT:
                    self.pose_quat = views[0].pose
                else:
                    self.pose = None

                frame = self.get_camera_image(cam_left_tex)
                if frame is not None:
                    frame_left = self.get_camera_image(cam_left_tex)
                    frame_right = self.get_camera_image(cam_right_tex)

                    frame_data_left = np.array(frame_left, np.uint8)
                    frame_data_right = np.array(frame_right, np.uint8)

                    GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id_left)
                    GL.glTexImage2D(
                        GL.GL_TEXTURE_2D,
                        0,
                        GL.GL_RGB,
                        frame_left.shape[1],
                        frame_left.shape[0],
                        0,
                        GL.GL_RGB,
                        GL.GL_UNSIGNED_BYTE,
                        frame_data_left,
                    )

                    GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id_right)
                    GL.glTexImage2D(
                        GL.GL_TEXTURE_2D,
                        0,
                        GL.GL_RGB,
                        frame_right.shape[1],
                        frame_right.shape[0],
                        0,
                        GL.GL_RGB,
                        GL.GL_UNSIGNED_BYTE,
                        frame_data_right,
                    )

                for view_index, view in enumerate(context.view_loop(frame_state)):
                    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
                    GL.glUseProgram(
                        shader_program
                    )  # Ensure the shader program is active

                    if view_index == 0:  # Left eye
                        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id_left)
                        GL.glUniform2f(
                            GL.glGetUniformLocation(shader_program, "offset"),
                            -self.image_offset,
                            0.0,
                        )
                    else:  # Right eye
                        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id_right)
                        GL.glUniform2f(
                            GL.glGetUniformLocation(shader_program, "offset"),
                            self.image_offset,
                            0.0,
                        )

                    GL.glBindVertexArray(VAO)
                    GL.glDrawElements(GL.GL_TRIANGLES, 6, GL.GL_UNSIGNED_INT, None)
                    GL.glBindVertexArray(0)
                    GL.glUseProgram(0)
                    GL.glDisable(GL.GL_BLEND)
                if context.session_state == xr.SessionState.FOCUSED:
                    self.poll_actions()
                    action_spaces = [
                        self.input.hand_space[Side.LEFT],
                        self.input.hand_space[Side.RIGHT],
                    ]
                    for index, space in enumerate(action_spaces):
                        space_location = xr.locate_space(
                            space=space,
                            base_space=context.space,
                            time=frame_state.predicted_display_time,
                        )
                        if (
                            space_location.location_flags
                            & xr.SPACE_LOCATION_POSITION_VALID_BIT
                        ):

                            if index == 0:
                                self.controller["left"] = space_location.pose
                            elif index == 1:
                                self.controller["right"] = space_location.pose
        exit()

    def get_camera_image(self, texture):
        while True:
            try:
                while not texture.hasRamImage():
                    sleep(0.01)
                break
            except:
                pass
            sleep(0.01)
        image = np.array(texture.getRamImageAs("RGB"), dtype=np.uint8)
        image = image.reshape((texture.getYSize(), texture.getXSize(), 3))
        if (texture.getXSize(), texture.getYSize()) != (2064, 2208):
            image = cv2.resize(
                image,
                (
                    texture.getXSize(),
                    texture.getYSize() * 2064 // 2208,
                ),
                interpolation=cv2.INTER_CUBIC,
            )
        return image

    def initialize_actions(self):
        # Create an action set.
        action_set_info = xr.ActionSetCreateInfo(
            action_set_name="gameplay",
            localized_action_set_name="Gameplay",
            priority=0,
        )
        self.input.action_set = xr.create_action_set(self.instance, action_set_info)
        # Get the XrPath for the left and right hands - we will use them as subaction paths.
        self.input.hand_subaction_path[Side.LEFT] = xr.string_to_path(
            self.instance, "/user/hand/left"
        )
        self.input.hand_subaction_path[Side.RIGHT] = xr.string_to_path(
            self.instance, "/user/hand/right"
        )
        # Create actions
        # Create an input action for grabbing objects with the left and right hands.
        self.input.grab_action = xr.create_action(
            action_set=self.input.action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.FLOAT_INPUT,
                action_name="grab_object",
                localized_action_name="Grab Object",
                count_subaction_paths=len(self.input.hand_subaction_path),
                subaction_paths=self.input.hand_subaction_path,
            ),
        )
        # Create an input action getting the left and right hand poses.
        self.input.pose_action = xr.create_action(
            action_set=self.input.action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.POSE_INPUT,
                action_name="hand_pose",
                localized_action_name="Hand Pose",
                count_subaction_paths=len(self.input.hand_subaction_path),
                subaction_paths=self.input.hand_subaction_path,
            ),
        )
        # Create output actions for vibrating the left and right controller.
        self.input.vibrate_action = xr.create_action(
            action_set=self.input.action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.VIBRATION_OUTPUT,
                action_name="vibrate_hand",
                localized_action_name="Vibrate Hand",
                count_subaction_paths=len(self.input.hand_subaction_path),
                subaction_paths=self.input.hand_subaction_path,
            ),
        )
        # Create input actions for quitting the session using the left and right controller.
        # Since it doesn't matter which hand did this, we do not specify subaction paths for it.
        # We will just suggest bindings for both hands, where possible.
        self.input.quit_action = xr.create_action(
            action_set=self.input.action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.BOOLEAN_INPUT,
                action_name="quit_session",
                localized_action_name="Quit Session",
                count_subaction_paths=0,
                subaction_paths=None,
            ),
        )

        select_path = [
            xr.string_to_path(self.instance, "/user/hand/left/input/select/click"),
            xr.string_to_path(self.instance, "/user/hand/right/input/select/click"),
        ]
        pose_path = [
            xr.string_to_path(self.instance, "/user/hand/left/input/grip/pose"),
            xr.string_to_path(self.instance, "/user/hand/right/input/grip/pose"),
        ]
        haptic_path = [
            xr.string_to_path(self.instance, "/user/hand/left/output/haptic"),
            xr.string_to_path(self.instance, "/user/hand/right/output/haptic"),
        ]
        menu_click_path = [
            xr.string_to_path(self.instance, "/user/hand/left/input/menu/click"),
            xr.string_to_path(self.instance, "/user/hand/right/input/menu/click"),
        ]
        trigger_value_path = [
            xr.string_to_path(self.instance, "/user/hand/left/input/trigger/value"),
            xr.string_to_path(self.instance, "/user/hand/right/input/trigger/value"),
        ]
        # Suggest bindings for KHR Simple.
        khr_bindings = [
            # Fall back to a click input for the grab action.
            xr.ActionSuggestedBinding(self.input.grab_action, select_path[Side.LEFT]),
            xr.ActionSuggestedBinding(self.input.grab_action, select_path[Side.RIGHT]),
            xr.ActionSuggestedBinding(self.input.pose_action, pose_path[Side.LEFT]),
            xr.ActionSuggestedBinding(self.input.pose_action, pose_path[Side.RIGHT]),
            xr.ActionSuggestedBinding(
                self.input.quit_action, menu_click_path[Side.LEFT]
            ),
            xr.ActionSuggestedBinding(
                self.input.quit_action, menu_click_path[Side.RIGHT]
            ),
            xr.ActionSuggestedBinding(
                self.input.vibrate_action, haptic_path[Side.LEFT]
            ),
            xr.ActionSuggestedBinding(
                self.input.vibrate_action, haptic_path[Side.RIGHT]
            ),
        ]
        xr.suggest_interaction_profile_bindings(
            instance=self.instance,
            suggested_bindings=xr.InteractionProfileSuggestedBinding(
                interaction_profile=xr.string_to_path(
                    self.instance,
                    "/interaction_profiles/khr/simple_controller",
                ),
                count_suggested_bindings=len(khr_bindings),
                suggested_bindings=(xr.ActionSuggestedBinding * len(khr_bindings))(
                    *khr_bindings
                ),
            ),
        )

        action_space_info = xr.ActionSpaceCreateInfo(
            action=self.input.pose_action,
            # pose_in_action_space # w already defaults to 1 in python...
            subaction_path=self.input.hand_subaction_path[Side.LEFT],
        )
        assert action_space_info.pose_in_action_space.orientation.w == 1
        self.input.hand_space[Side.LEFT] = xr.create_action_space(
            session=self.session,
            create_info=action_space_info,
        )
        action_space_info.subaction_path = self.input.hand_subaction_path[Side.RIGHT]
        self.input.hand_space[Side.RIGHT] = xr.create_action_space(
            session=self.session,
            create_info=action_space_info,
        )

        xr.attach_session_action_sets(
            session=self.session,
            attach_info=xr.SessionActionSetsAttachInfo(
                count_action_sets=1,
                action_sets=pointer(self.input.action_set),
            ),
        )

    def update_image_offset(self, new_offset):
        self.image_offset += new_offset

    def set_image_offset(self, new_offset):
        self.image_offset = new_offset

    def compile_shader(self, source, shader_type):
        shader = GL.glCreateShader(shader_type)
        GL.glShaderSource(shader, source)
        GL.glCompileShader(shader)
        if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
            raise RuntimeError(GL.glGetShaderInfoLog(shader))
        return shader

    def poll_actions(self) -> None:
        """Sample input actions and generate haptic feedback."""
        self.input.hand_active[:] = [xr.FALSE, xr.FALSE]

        # Sync actions
        active_action_set = xr.ActiveActionSet(self.input.action_set, xr.NULL_PATH)
        xr.sync_actions(
            self.session,
            xr.ActionsSyncInfo(
                count_active_action_sets=1,
                active_action_sets=pointer(active_action_set),
            ),
        )
        for hand in Side:
            grab_value: xr.ActionStateFloat = xr.get_action_state_float(
                session=self.session,
                get_info=xr.ActionStateGetInfo(
                    action=self.input.grab_action,
                    subaction_path=self.input.hand_subaction_path[hand],
                ),
            )
            if grab_value.is_active:
                if (
                    grab_value.current_state
                    > self.HandControl.hands[hand].haptic_threshold
                ):
                    vibration = xr.HapticVibration(
                        amplitude=self.HandControl.hands[hand].haptic_strength,
                        duration=xr.MIN_HAPTIC_DURATION,
                        frequency=self.HandControl.hands[hand].haptic_frequency,
                    )
                    xr.apply_haptic_feedback(
                        session=self.session,
                        haptic_action_info=xr.HapticActionInfo(
                            action=self.input.vibrate_action,
                            subaction_path=self.input.hand_subaction_path[hand],
                        ),
                        haptic_feedback=cast(
                            byref(vibration), POINTER(xr.HapticBaseHeader)
                        ).contents,
                    )
            pose_state = xr.get_action_state_pose(
                session=self.session,
                get_info=xr.ActionStateGetInfo(
                    action=self.input.pose_action,
                    subaction_path=self.input.hand_subaction_path[hand],
                ),
            )
            self.input.hand_active[hand] = pose_state.is_active
            self.HandControl.hands[hand].active = pose_state.is_active
            self.HandControl.hands[hand].trigger_value = grab_value.current_state

    def create_shader_program(self, vertex_source, fragment_source):
        vertex_shader = self.compile_shader(vertex_source, GL.GL_VERTEX_SHADER)
        fragment_shader = self.compile_shader(fragment_source, GL.GL_FRAGMENT_SHADER)
        program = GL.glCreateProgram()
        GL.glAttachShader(program, vertex_shader)
        GL.glAttachShader(program, fragment_shader)
        GL.glLinkProgram(program)
        if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
            raise RuntimeError(GL.glGetProgramInfoLog(program))
        GL.glDeleteShader(vertex_shader)
        GL.glDeleteShader(fragment_shader)
        return program

    def haptic_feedback(
        self, hand: int, amplitude: float, duration: int, frequency: int
    ):
        vibration = xr.HapticVibration(
            amplitude=amplitude,
            duration=duration,
            frequency=frequency,
        )
        xr.apply_haptic_feedback(
            session=self.session,
            haptic_action_info=xr.HapticActionInfo(
                action=self.input.vibrate_action,
                subaction_path=self.input.hand_subaction_path[hand],
            ),
            haptic_feedback=cast(
                byref(vibration), POINTER(xr.HapticBaseHeader)
            ).contents,
        )

    def initialize_audio(self):
        """Initialize audio settings."""
        try:
            self.audio = pyaudio.PyAudio()

            # Find the VR headset audio device
            device_index = None
            for i in range(self.audio.get_device_count()):
                device_info = self.audio.get_device_info_by_index(i)
                if "Oculus" in device_info.get("name", ""):
                    device_index = i
                    break

            if device_index is None:
                print("VR headset audio device not found.")
                return

            # Open an audio stream with the VR headset as the output device
            self.audio_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=2,
                rate=44100,
                output=True,
                output_device_index=device_index,
            )

            print(
                f"Initialized VR headset audio device: {self.audio.get_device_info_by_index(device_index).get('name')}"
            )
        except Exception as e:
            print(f"An error occurred while initializing audio: {e}")

    def close_audio(self):
        """Close the audio stream and terminate PyAudio."""
        if hasattr(self, "audio_stream"):
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if hasattr(self, "audio"):
            self.audio.terminate()


main = main()
if __name__ == "__main__":
    BaseVrApp().run()
