from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing
from omni.isaac.manipulators.grippers.gripper import Gripper


class PlaceController(BaseController):
    """
    A state machine controller for placing an object.

    This controller follows a sequence of phases to place an object.

    The phases are:
    - Phase 0: Move end effector to pour position.
    - Phase 1: Move end effector to return position.
    - Phase 2: Pause at return position.
    - Phase 3: Open gripper.
    - Phase 4: Move end effector to initial height.

    Args:
        name (str): Identifier for the controller.
        cspace_controller (BaseController): A cartesian space controller returning an ArticulationAction type.
        gripper (Gripper): A gripper controller for open/close actions.
        end_effector_initial_height (float, optional): Initial height for the end effector. Defaults to 0.3 meters.
        events_dt (list of float, optional): Time duration for each phase. Defaults to [0.005, 0.005, 0.005, 0.1, 0.005] divided by speed if not specified.
        speed (float, optional): Speed multiplier for phase durations. Defaults to 1.0.

    Raises:
        Exception: If 'events_dt' is not a list or numpy array.
        Exception: If 'events_dt' length is greater than 5.
    """

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
        speed: float = 1.0
    ) -> None:
        super().__init__(name=name)
        self._event = 0
        self._t = 0
        self._h1 = end_effector_initial_height
        if self._h1 is None:
            self._h1 = 0.32 / get_stage_units()
        self._h0 = None
        self._events_dt = events_dt
        if events_dt is None:
            self._events_dt = [dt / speed for dt in [0.005, 0.005, 0.005, 0.1, 0.005]]
        else:
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 5:
                raise Exception("events dt length must be less than 5")

        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self._pause = False
        return

    def is_paused(self) -> bool:
        """
        Check if the state machine is paused.

        Returns:
            bool: True if paused, False otherwise.
        """
        return self._pause

    def get_current_event(self) -> int:
        """
        Get the current phase/event of the state machine.

        Returns:
            int: Current phase/event number.
        """
        return self._event

    def forward(
        self,
        pour_position: np.ndarray,
        return_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: typing.Optional[np.ndarray] = None,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """
        Execute one step of the controller.

        Args:
            pour_position (np.ndarray): Position to move the end effector for pouring.
            return_position (np.ndarray): Position to move the end effector for returning.
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            end_effector_offset (np.ndarray, optional): Offset of the end effector target. Defaults to None.
            end_effector_orientation (np.ndarray, optional): Orientation of the end effector. Defaults to None.

        Returns:
            ArticulationAction: Action to be executed by the ArticulationController.
        """
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0])
        if self._pause or self.is_done():
            self.pause()
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)

        if self._event == 3:
            # Open the gripper
            target_joint_positions = self._gripper.forward(action="open")
        else:
            if self._event == 0:
                self._current_target_x = pour_position[0]
                self._current_target_y = pour_position[1]
                self._h0 = pour_position[2]

            interpolated_xy = self._get_interpolated_xy(
                return_position[0], return_position[1], self._current_target_x, self._current_target_y
            )

            if self._event == 1:
                target_position2 = return_position + np.array([0, 0, self._h1]) - np.array([0, 0, return_position[2]])
                target_joint_positions = self._cspace_controller.forward(
                    target_end_effector_position=target_position2,
                    target_end_effector_orientation=end_effector_orientation
                )
            else:
                target_height = self._get_target_hs(return_position[2])
                position_target = np.array(
                    [
                        interpolated_xy[0] + end_effector_offset[0],
                        interpolated_xy[1] + end_effector_offset[1],
                        target_height + end_effector_offset[2],
                    ]
                )
                if end_effector_orientation is None:
                    end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
                target_joint_positions = self._cspace_controller.forward(
                    target_end_effector_position=position_target,
                    target_end_effector_orientation=end_effector_orientation
                )

        if self._event == 1:
            # Check if the current position has reached the target position
            current_positions = np.array(self._gripper.get_world_pose())[0]
            target_position2 = return_position + np.array([0, 0, self._h1]) - np.array([0, 0, return_position[2]])
            if np.allclose(current_positions, target_position2, atol=0.05):
                self._event += 1
        else:
            self._t += self._events_dt[self._event]
            if self._t >= 1.0:
                self._event += 1
                self._t = 0

        return target_joint_positions

    def _get_interpolated_xy(self, target_x, target_y, current_x, current_y):
        """
        Interpolate the x and y coordinates between the current and target positions.

        Args:
            target_x (float): Target x position.
            target_y (float): Target y position.
            current_x (float): Current x position.
            current_y (float): Current y position.

        Returns:
            np.ndarray: Interpolated x and y coordinates.
        """
        alpha = self._get_alpha()
        xy_target = (1 - alpha) * np.array([current_x, current_y]) + alpha * np.array([target_x, target_y])
        return xy_target

    def _get_alpha(self):
        """
        Calculate the interpolation factor based on the current phase.

        Returns:
            float: Interpolation factor.
        """
        if self._event == 0:
            return 0
        elif self._event == 1:
            return self._mix_sin(self._t)
        elif self._event in [2, 3, 4]:
            return 1.0
        else:
            raise ValueError()

    def _get_target_hs(self, target_height):
        """
        Calculate the target height for the end effector based on the current phase.

        Args:
            target_height (float): Target height for the end effector.

        Returns:
            float: Calculated target height.
        """
        if self._event == 0:
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(self._h0, self._h1, a)
        elif self._event == 1:
            h = self._h1
        elif self._event == 2:
            h = self._combine_convex(self._h1, target_height, self._mix_sin(self._t))
        elif self._event == 3:
            h = target_height
        elif self._event == 4:
            h = self._combine_convex(target_height, self._h1, self._mix_sin(self._t))
        else:
            raise ValueError()
        return h

    def _mix_sin(self, t):
        """
        Calculate the interpolation factor using a sine function for smooth transitions.

        Args:
            t (float): Time parameter.

        Returns:
            float: Interpolation factor.
        """
        return 0.5 * (1 - np.cos(t * np.pi))

    def _combine_convex(self, a, b, alpha):
        """
        Perform convex combination of two values.

        Args:
            a (float): First value.
            b (float): Second value.
            alpha (float): Interpolation factor.

        Returns:
            float: Convex combination of the two values.
        """
        return (1 - alpha) * a + alpha * b

    def reset(
        self,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        """
        Reset the state machine to start from the first phase.

        Args:
            end_effector_initial_height (float, optional): Initial height for the end effector. Defaults to None.
            events_dt (list of float, optional): Time duration for each phase. Defaults to None.

        Raises:
            Exception: If 'events_dt' is not a list or numpy array.
            Exception: If 'events_dt' length is greater than 5.
        """
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        if end_effector_initial_height is not None:
            self._h1 = end_effector_initial_height
        self._pause = False
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("event velocities need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 5:
                raise Exception("events dt length must be less than 5")
        return

    def is_done(self) -> bool:
        """
        Check if the state machine has reached the last phase.

        Returns:
            bool: True if the last phase is reached, False otherwise.
        """
        return self._event >= len(self._events_dt)

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
    
