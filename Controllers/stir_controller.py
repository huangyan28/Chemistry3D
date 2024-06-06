from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing

class StirController(BaseController):
    """
    A stir controller that makes the robotic arm move in a circular motion at a fixed height.

    This controller performs a stirring motion with the specified radius and number of rotations.

    Args:
        name (str): Identifier for the controller.
        cspace_controller (BaseController): A cartesian space controller returning an ArticulationAction type.
        fixed_height (float, optional): Height at which to perform the circular motion. Defaults to 0.3 meters if not specified.
        radius (float, optional): Radius of the circular motion. Defaults to 0.1 meters if not specified.
        speed (float, optional): Speed of the circular motion in radians per second. Defaults to 1.0.

    Raises:
        Exception: If any parameter is invalid.
    """

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        fixed_height: typing.Optional[float] = None,
        radius: float = 0.1,
        speed: float = 1.0
    ) -> None:
        BaseController.__init__(self, name=name)
        self._t = 0
        self._height = fixed_height if fixed_height is not None else 0.3 / get_stage_units()
        self._radius = radius
        self._speed = speed
        self._cspace_controller = cspace_controller
        self._pause = False
        self._start = True
        self._total_time = 0
        self._num_rotations = 0
        return

    def is_paused(self) -> bool:
        """
        Check if the state machine is paused.

        Returns:
            bool: True if paused, False otherwise.
        """
        return self._pause

    def forward(
        self,
        center_position: np.ndarray,
        current_joint_positions: np.ndarray,
        num_rotations: int,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """
        Execute one step of the controller.

        Args:
            center_position (np.ndarray): Center position of the circular motion.
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            num_rotations (int): Number of rotations to perform.
            end_effector_orientation (np.ndarray, optional): Orientation of the end effector. Defaults to None.

        Returns:
            ArticulationAction: Action to be executed by the ArticulationController.
        """
        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))

        if self._start:
            self._start = False
            self._t = 0
            self._total_time = 0
            self._num_rotations = num_rotations

        if self._pause or self.is_done():
            self.pause()
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)

        # Calculate the target position for circular motion
        x_offset = self._radius * np.cos(self._speed * self._t)
        y_offset = self._radius * np.sin(self._speed * self._t)
        position_target = np.array([center_position[0] + x_offset, center_position[1] + y_offset, self._height])

        target_joint_positions = self._cspace_controller.forward(
            target_end_effector_position=position_target,
            target_end_effector_orientation=end_effector_orientation
        )
        self._t += 0.01  # Increment time for the next step
        self._total_time += 0.01

        # Check if the desired number of rotations is reached
        if self._total_time >= (2 * np.pi * num_rotations) / self._speed:
            self._pause = True  # Indicate that the motion is done

        return target_joint_positions

    def reset(self, fixed_height: typing.Optional[float] = None, radius: float = 0.1, speed: float = 1.0) -> None:
        """
        Reset the state machine to start from the initial phase.

        Args:
            fixed_height (float, optional): Height at which to perform the circular motion. Defaults to None.
            radius (float, optional): Radius of the circular motion. Defaults to 0.1 meters.
            speed (float, optional): Speed of the circular motion in radians per second. Defaults to 1.0.
        """
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._t = 0
        self._total_time = 0
        if fixed_height is not None:
            self._height = fixed_height
        self._radius = radius
        self._speed = speed
        self._pause = False
        self._start = True
        return

    def is_done(self) -> bool:
        """
        Check if the state machine has completed the circular motion.

        Returns:
            bool: True if the desired number of rotations is completed, False otherwise.
        """
        return self._total_time >= (2 * np.pi * self._num_rotations) / self._speed

    def pause(self) -> None:
        """
        Pause the state machine's time and phase.
        """
        self._pause = True
        return

    def resume(self) -> None:
        """
        Resume the state machine's time and phase.
        """
        self._pause = False
        return
