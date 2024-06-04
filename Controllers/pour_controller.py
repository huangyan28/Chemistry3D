from omni.isaac.core.controllers import BaseController
from omni.isaac.core.controllers.articulation_controller import ArticulationController
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators.grippers.gripper import Gripper
from pxr import Gf
import numpy as np
import typing
from Sim_Container import Sim_Container
from utils import *

class PourController(BaseController):
    """
    A state machine for performing a pouring action with a gripper.

    This controller follows a sequence of states to perform a pouring action.

    The states are:
    - State 0: Start pouring (e.g., tilt the gripper forward).
    - State 1: Pause for a moment.
    - State 2: Stop pouring (e.g., tilt the gripper back to the original position).

    Args:
        name (str): Identifier for the controller.
        cspace_controller (BaseController): A cartesian space controller returning an ArticulationAction type.
        gripper (Gripper): A gripper controller for open/close actions.
        Sim_Container1 (Sim_Container, optional): The first simulation container involved in the pouring action.
        Sim_Container2 (Sim_Container, optional): The second simulation container involved in the pouring action.
        pour_volume (int, optional): The volume of liquid to pour.
        events_dt (list of float, optional): Time duration for each phase. Defaults to [0.004, 0.002, 0.004] divided by speed if not specified.
        speed (float, optional): Speed multiplier for phase durations. Defaults to 1.0.

    Raises:
        Exception: If 'events_dt' is not a list or numpy array.
        Exception: If 'events_dt' length is greater than 3.
    """

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper,
        Sim_Container1: Sim_Container = None,
        Sim_Container2: Sim_Container = None,
        pour_volume: int = None,
        events_dt: typing.Optional[typing.List[float]] = None,
        speed: float = 1.0
    ) -> None:
        BaseController.__init__(self, name=name)
        self.Sim_Container1 = Sim_Container1
        self.Sim_Container2 = Sim_Container2
        self.pour_volume = pour_volume
        self._forward_start = False
        self._event = 0
        self._t = 0
        self._events_dt = events_dt
        if self._events_dt is None:
            self._events_dt = [dt / speed for dt in [0.004, 0.002, 0.004]]
        else:
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 3:
                raise Exception("events dt need have length of 3 or less")
        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self._pause = False
        self._start = True
        self._pour_default_speed = -60 / 180.0 * np.pi  # Example pour speed in radians per second
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

    def iter_added(self):
        """
        Get the current value of _iter_added.

        Returns:
            bool: The current value of _iter_added.
        """
        return self._iter_added

    def set_iter_added(self, value):
        """
        Set the value of _iter_added.

        Args:
            value (bool): The value to set for _iter_added.

        Raises:
            ValueError: If the value is not a boolean.
        """
        if isinstance(value, bool):
            self._iter_added = value
        else:
            raise ValueError("Value must be a boolean")

    def able_to_react(self) -> bool:
        """
        Check if the containers are ready to react.

        Returns:
            bool: True if containers are ready, False otherwise.
        """
        if self.Sim_Container1.get_info()[0] and self.Sim_Container2.get_info()[0] is None:
            return False
        else:
            return True

    def get_current_info(self):
        """
        Get the current information from Sim_Container2.

        Returns:
            The current information from Sim_Container2.
        """
        return self.Sim_Container2.get_info()

    def update_sim_container_color(self):
        """
        Update the color of the liquid in the simulation containers.
        """
        if self.Sim_Container1.org:
            return
        else:
            color = self.get_new_liquid_color()
            update_color = Gf.Vec3f(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
            for shader in self.Sim_Container1.get_object()['liquid']:
                material_color_set(shader['material_shader'], update_color)
            for shader in self.Sim_Container2.get_object()['liquid']:
                material_color_set(shader['material_shader'], update_color)
            return

    def check_need_solid_melt(self):
        """
        Check if solid needs to be melted.

        Returns:
            bool: True if solid needs to be melted, False otherwise.
        """
        if not self._solid_melted:
            return bool(self.Sim_Container2.get_object()['solid'] is not None)
        else:
            return False

    def get_concentrations_iter(self):
        """
        Get the iterator for concentrations.

        Returns:
            iterator: The concentrations iterator.
        """
        return self._concentrations_iter

    def check_need_for_new_liquid(self):
        """
        Check if a new liquid needs to be created in Sim_Container2.

        Returns:
            bool: True if a new liquid needs to be created, False otherwise.
        """
        if not self._liquid_created:
            has_liquid_in_container1 = bool(self.Sim_Container1.get_object()['liquid'])
            has_liquid_in_container2 = bool(self.Sim_Container2.get_object()['liquid'])
            return has_liquid_in_container1 and not has_liquid_in_container2
        else:
            return False

    def get_new_liquid_color(self):
        """
        Get the color of the new liquid in Sim_Container2.

        Returns:
            The color of the new liquid.
        """
        return self.Sim_Container2.get_color()

    def Update_Solute(self):
        """
        Update the solute by transferring liquids and solids from Sim_Container1 to Sim_Container2.
        """
        if self._liquid_created:
            return

        objects_1 = self.Sim_Container1.get_object()
        for liquid in objects_1['liquid']:
            self.Sim_Container2.add_liquid(liquid)
        for solid in objects_1['solid']:
            self.Sim_Container2.add_solid(solid)
        self.Sim_Container1.set_object({'liquid': [], 'solid': []})
        return

    def reaction_started(self):
        """
        Check if the reaction has started by evaluating the contact between centroids.

        Returns:
            bool: True if the reaction has started, False otherwise.
        """
        centroid1 = None
        centroid2 = None

        if self.Sim_Container1.has_liquid:
            centroid1 = get_ParticleSet_Centroid(particle_set=self.Sim_Container1.get_object()['liquid'][0]['particle_set'])
        elif self.Sim_Container1.has_solid:
            centroid1, _ = self.Sim_Container1.get_object()['solid'][0].get_world_pose()

        if not self._reaction_started:
            if self.Sim_Container2.has_liquid:
                centroid2 = get_ParticleSet_Centroid(particle_set=self.Sim_Container2.get_object()['liquid'][0]['particle_set'])
            elif self.Sim_Container2.has_solid:
                centroid2, _ = self.Sim_Container2.get_object()['solid'][0].get_world_pose()
            if centroid1 is not None and centroid2 is not None:
                self._reaction_started = is_contact(centroid1, centroid2, 0.02)

        return self._reaction_started

    def _get_sim_container1(self):
        """
        Get Sim_Container1.

        Returns:
            Sim_Container: Sim_Container1.
        """
        return self.Sim_Container1
    
    def _get_sim_container2(self):
        """
        Get Sim_Container2.

        Returns:
            Sim_Container: Sim_Container2.
        """
        return self.Sim_Container2

    def forward(
        self,
        franka_art_controller: ArticulationController,
        current_joint_positions: np.ndarray,
        current_joint_velocities: np.ndarray,
        pour_speed: float = None,
    ) -> ArticulationAction:
        """
        Execute one step of the controller.

        Args:
            franka_art_controller (ArticulationController): The articulation controller for the Franka robot.
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            current_joint_velocities (np.ndarray): Current joint velocities of the robot.
            pour_speed (float, optional): Speed for the pouring action. Defaults to None.

        Returns:
            ArticulationAction: Action to be executed by the ArticulationController.
        """
        if self._forward_start is False:
            self.concentrations_list = self.Sim_Container2.update(self.Sim_Container1, self.pour_volume)
            self._concentrations_iter = iter(self.concentrations_list)
            self._iter_added = False
            self._liquid_created = False
            self.need_solid_melt = False
            self._solid_melted = False
            self._reaction_started = False
            self._forward_start = True
            self.Sim_Container1.has_liquid = len(self.Sim_Container1.get_object()['liquid']) > 0
            self.Sim_Container1.has_solid = len(self.Sim_Container1.get_object()['solid']) > 0
            self.Sim_Container2.has_liquid = len(self.Sim_Container2.get_object()['liquid']) > 0
            self.Sim_Container2.has_solid = len(self.Sim_Container2.get_object()['solid']) > 0
        
        if pour_speed is None:
            self._pour_speed = self._pour_default_speed
        else:
            self._pour_speed = pour_speed
            
        if self._pause or self._event >= len(self._events_dt):
            target_joints = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joints)
            
        if self._event == 0 and self._start:
            self._start = False
            self._target_end_effector_orientation = self._gripper.get_world_pose()[1]

        if self._event == 0:
            franka_art_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[6] = self._pour_speed
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)

        if self._event == 1:
            franka_art_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[6] = 0
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)
            
        if self._event == 2:
            franka_art_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[6] = -self._pour_speed
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)

        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0

        return target_joints

    def reset(self, events_dt: typing.Optional[typing.List[float]] = None) -> None:
        """
        Reset the state machine to start from the first phase.

        Args:
            events_dt (list of float, optional): Time duration for each phase. Defaults to None.

        Raises:
            Exception: If 'events_dt' is not a list or numpy array.
            Exception: If 'events_dt' length is greater than 3.
        """
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        self._pause = False
        self._start = True
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 3:
                raise Exception("events dt need have length of 3 or less")
        return

    def is_done(self) -> bool:
        """
        Check if the state machine has reached the last phase.

        Returns:
            bool: True if the last phase is reached, False otherwise.
        """
        if self._event >= len(self._events_dt):
            self.Update_Solute()
            print(f'Sim_Container1 objects: {self.Sim_Container1.get_object()}')
            print(f'Sim_Container2 objects: {self.Sim_Container2.get_object()}')
            return True
        else:
            return False

    def is_poured(self) -> bool:
        """
        Check if the pouring action has completed.

        Returns:
            bool: True if the pouring action has completed, False otherwise.
        """
        return self._event >= 2

    def started_pouring(self) -> bool:
        """
        Check if the pouring action has started.

        Returns:
            bool: True if the pouring action has started, False otherwise.
        """
        return self._event >= 1
    
    def solid_melted(self) -> bool:
        """
        Check if the solid has melted.

        Returns:
            bool: True if the solid has melted, False otherwise.
        """
        return self._solid_melted

    def solid_melted_set(self, value: bool) -> None:
        """
        Set the solid melted state.

        Args:
            value (bool): The state to set for solid melted.
        """
        self._solid_melted = value
        return
    
    def liquid_created(self) -> bool:
        """
        Check if the liquid has been created.

        Returns:
            bool: True if the liquid has been created, False otherwise.
        """
        return self._liquid_created
    
    def liquid_created_set(self, value: bool) -> None:
        """
        Set the liquid created state.

        Args:
            value (bool): The state to set for liquid created.
        """
        self._liquid_created = value
        return

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
