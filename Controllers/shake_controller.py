class ShakeController(BaseController):
    """
    A state machine for performing a shaking action with a gripper.

    Each phase runs for a specified duration, defined by events_dt.

    - State 0: Shake forward (e.g., move the gripper forward).
    - State 1: Shake backward (e.g., move the gripper backward).
    - State 2: Shake backward (e.g., move the gripper backward).
    - State 3: Shake forward (e.g., move the gripper forward).

    Args:
        name (str): Name ID of the controller.
        cspace_controller (BaseController): A Cartesian space controller that returns an ArticulationAction type.
        gripper (Gripper): A gripper controller for open/close actions.
        events_dt (typing.Optional[typing.List[float]], optional): Duration of each phase. Defaults to None.
        speed (float): Speed of the shake action. Defaults to 1.0.

    Raises:
        Exception: If events_dt is not a list or numpy array.
        Exception: If events_dt has a length greater than 4.
    """

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper,
        events_dt: typing.Optional[typing.List[float]] = None,
        speed: float = 1.0
    ) -> None:
        BaseController.__init__(self, name=name)
        self._shake_start = False
        self._event = 0
        self._t = 0
        self._events_dt = events_dt
        if self._events_dt is None:
            self._events_dt = [dt / speed for dt in [0.1, 0.1, 0.1, 0.1]]
        else:
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 4:
                raise Exception("events dt need have length of 4 or less")
        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self._pause = False
        self._start = True
        self._shake_speed = 180 / 180.0 * np.pi  # Example shake speed in radians per second
        return

    def is_paused(self) -> bool:
        """
        Returns:
            bool: True if the state machine is paused. Otherwise, False.
        """
        return self._pause

    def get_current_event(self) -> int:
        """
        Returns:
            int: Current event/phase of the state machine.
        """
        return self._event

    def forward(
        self,
        franka_art_controller: ArticulationController,
        current_joint_positions: np.ndarray,
        current_joint_velocities: np.ndarray,
        num_shakes: int = 1,
        shake_speed: float = None,
    ) -> ArticulationAction:
            
        if self._shake_start is False:
            self._num_shakes = num_shakes
            self._shake_start = True
            self.current_shake = 0
        
        if shake_speed is None:
            self._shake_speed = self._shake_speed
        else:
            self._shake_speed = shake_speed
            
        if self._pause or self.current_shake >= num_shakes:
            target_joints = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joints)
            
        if self._event == 0 and self._start:
            self._start = False

        if self._event == 0:
            franka_art_controller.switch_dof_control_mode(dof_index=0, mode="velocity")
            # Example: move the gripper forward
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[6] = self._shake_speed
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)

        if self._event == 1:
            franka_art_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            # Example: move the gripper backward
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[6] = -self._shake_speed
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)
            
        if self._event == 2:
            franka_art_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[6] = -self._shake_speed
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)  
            
        if self._event == 3:
            franka_art_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[6] = self._shake_speed
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)  

        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1 
            if self._event >= 4:
                self._event = 0
                self.current_shake += 1
            self._t = 0

        return target_joints

    def reset(self, events_dt: typing.Optional[typing.List[float]] = None) -> None:
        """
        Resets the state machine to its initial state.

        Args:
            events_dt (typing.Optional[typing.List[float]], optional): Duration of each phase. Defaults to None.
        """
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        self._pause = False
        self._start = True
        self.current_shake = 0
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 4:
                raise Exception("events dt need have length of 4 or less")
        return

    def is_done(self) -> bool:
        """
        Returns:
            bool: True if the state machine has completed all phases. Otherwise, False.
        """
        return self.current_shake >= self._num_shakes

    def is_shaking(self) -> bool:
        """
        Returns:
            bool: True if the state machine is currently in a shaking phase. Otherwise, False.
        """
        return self._event < 3 and self.current_shake < self._num_shakes

    def pause(self) -> None:
        """
        Pauses the state machine's time and phase.
        """
        self._pause = True
        return

    def resume(self) -> None:
        """
        Resumes the state machine's time and phase.
        """
        self._pause = False
        return
