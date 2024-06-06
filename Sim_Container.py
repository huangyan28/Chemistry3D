from chem_sim.simulation.simulator import Container
from utils import Utils
import numpy as np
from pxr import Gf,Sdf
from Controllers.Controller_Manager import ControllerManager
from Controllers.pick_move_controller import PickMoveController
from Controllers.return_controller import PlaceController
from utils import Utils
from utils import *
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from pxr import Sdf, Gf, UsdPhysics
import omni.isaac.core.utils.prims as prims_utils
from omni.isaac.core.objects import DynamicCuboid

utils = Utils()
scenePath = Sdf.Path("/physicsScene")

class Sim_Container(Container):
    def __init__(self, world, sim_container, object=None, solute=None, org=False, volume=0, temp=25, verbose=False):
        # Initialize the Container base class
        super().__init__(solute, org, volume, temp, verbose)
        if org is False and solute is not None:
            self.update(self)

        self.world = world
        self.sim_container = sim_container
        # Ensure 'object' is a dictionary containing two lists (liquid and solid)
        rbApi = UsdPhysics.RigidBodyAPI.Apply(self.sim_container.prim.GetPrim())
        rbApi.CreateRigidBodyEnabledAttr(True)
        current_observations = self.world.get_observations()
        
        if solute is not None and volume != 's':
            utils._set_particle_parameter(self.world, particleContactOffset=0.003)
            print(f"/World/particleSystem{self.sim_container.name}")
            particle_system_set_material_dict = utils.create_particle_system_and_set(
                self.world,
                particle_system_path_str=f"/World/particleSystem{self.sim_container.name}",
                particle_set_path_str=f"/World/particles{self.sim_container.name}",
                scenePath=scenePath,
                center=Gf.Vec3f(current_observations[self.sim_container.name]["Default_Position"][0], 
                                current_observations[self.sim_container.name]["Default_Position"][1],  
                                current_observations[self.sim_container.name]["Default_Position"][2] + 0.01),
                dim_x=10,
                dim_y=10,
                dim_z=6,
                material_color=self.get_color()
            )
            self.object = {'liquid': [particle_system_set_material_dict], 'solid': []}

        if solute is not None and volume == 's':
            solid = world.scene.add( 
                DynamicCuboid(
                    prim_path=f"/World/Solid_{self.sim_container.name}",  # The prim path of the cube in the USD stage
                    name=f"Solid_{self.sim_container.name}",  # The unique name used to retrieve the object from the scene later on
                    position=np.array([current_observations[self.sim_container.name]["Default_Position"][0], 
                                       current_observations[self.sim_container.name]["Default_Position"][1],  
                                       current_observations[self.sim_container.name]["Default_Position"][2] + 0.01]),  # Using the current stage units which is in meters by default
                    scale=np.array([0.01, 0.01, 0.01]),  # Most arguments accept mainly numpy arrays
                    color=np.array(self.get_color())
                ))
            self.object = {'liquid': [], 'solid': [solid]}

        if solute is None:
            self.object = {'liquid': [], 'solid': []}

    def get_color(self):
        """Retrieve the color based on the organic property and solute information."""
        if self.org:
            return Gf.Vec3f(255, 255, 255)
        else:
            if self.get_info()[1]['color'][:3] == [0, 0, 0]:
                return [255, 255, 255]
            return self.get_info()[1]['color'][:3]
        
    def sim_update(self, Sim_Container1, robot, controller_manager, pour_volume=None):
        """Update simulation with the new state and task details for the controller."""
        pickmove_controller = PickMoveController(
            name="pickmove_controller",
            cspace_controller=RMPFlowController(name="pickmove_cspace_controller", robot_articulation=robot),
            gripper=robot.gripper,
            speed=1.5
        )

        from Controllers.pour_controller import PourController
        pour_controller = PourController(
            name="pour_controller",
            cspace_controller=RMPFlowController(name="pour_cspace_controller", robot_articulation=robot),
            gripper=robot.gripper,
            Sim_Container1=Sim_Container1,
            Sim_Container2=self,
            pour_volume=pour_volume
        )

        return_controller = PlaceController(
            name="return_controller",
            cspace_controller=RMPFlowController(name="return_cspace_controller", robot_articulation=robot),
            gripper=robot.gripper,
            speed=1.5
        )

        controller_manager.add_controller('pickmove_controller', pickmove_controller)
        controller_manager.add_controller('pour_controller', pour_controller)
        controller_manager.add_controller('return_controller', return_controller)

        controller_manager.add_task("pick", {
            "picking_position": lambda obs: obs[Sim_Container1.get_sim_container().name]["position"],
            "target_position": lambda obs: obs[Sim_Container1.get_sim_container().name]["Pour_Position"],
            "current_joint_positions": lambda obs: robot.get_joint_positions(),
            "end_effector_offset": np.array([0.0, 0.0, 0.06]),
            "end_effector_orientation": euler_angles_to_quat(np.array([np.pi / 2, np.pi / 2, 0]))
        })
        
        controller_manager.add_task("pour", {
            'franka_art_controller': robot.get_articulation_controller(),
            "current_joint_positions": robot.get_joint_positions(),
            'current_joint_velocities': robot.get_joint_velocities(),
            'pour_speed': lambda obs: obs[Sim_Container1.get_sim_container().name]["Pour_Derecition"] * 55 / 180.0 * np.pi
        })
        
        controller_manager.add_task("return", {
            "pour_position": lambda obs: obs[Sim_Container1.get_sim_container().name]["Pour_Position"],
            "return_position": lambda obs: obs[Sim_Container1.get_sim_container().name]["Return_Position"],
            "current_joint_positions": lambda obs: robot.get_joint_positions(),
            "end_effector_offset": np.array([0.0, 0.00, 0.055]),
            "end_effector_orientation": euler_angles_to_quat(np.array([np.pi / 2, np.pi / 2, 0]))
        })

    def set_sim_container(self, new_sim_container):
        """Set the simulation container."""
        self.sim_container = new_sim_container

    def get_sim_container(self):
        """Get the current simulation container."""
        return self.sim_container
    
    def set_object(self, new_object):
        """Update the object dictionary."""
        self.object = new_object

    def get_object(self):
        """Get the current object dictionary."""
        return self.object

    def add_liquid(self, liquid):
        """Add a liquid to the liquid list."""
        self.object['liquid'].append(liquid)
    
    def remove_solid(self, solid):
        """Remove a solid from the solid list."""
        self.object['solid'].clear()
    
    def melt_solid(self, controller_manager):
        """Melt and remove solid objects."""
        for solid in self.object['solid']:
            prim_path = solid.prim.GetPath().pathString
            prims_utils.delete_prim(prim_path)
            self.remove_solid(solid)
        controller_manager.set_need_solid_melt(False)

    def create_liquid(self, controller_manager, current_observations):
        """Create a new liquid in the simulation."""
        particle_system_set_material_dict = utils.create_particle_system_and_set(
            self.world,
            particle_system_path_str=f"/World/particelsystem_{controller_manager.get_current_controller_name()}",
            particle_set_path_str=f"/World/particles_{controller_manager.get_current_controller_name()}",
            scenePath=scenePath,
            center=Gf.Vec3f(
                float(current_observations[self.get_sim_container().name]["position"][0]),
                float(current_observations[self.get_sim_container().name]["Default_Position"][1]),
                float(current_observations[self.get_sim_container().name]["Default_Position"][2]) + 0.01
            ),
            dim_x=10,
            dim_y=10,
            dim_z=3,
            material_color=controller_manager.get_current_liquid_color()
        )
        self.add_liquid(particle_system_set_material_dict)
        controller_manager.set_need_new_liquid(False)

    def remove_liquid(self, liquid):
        """Remove a liquid from the liquid list if it exists."""
        if liquid in self.object['liquid']:
            self.object['liquid'].remove(liquid)

    def add_solid(self, solid):
        """Add a solid to the solid list."""
        self.object['solid'].append(solid)

    def remove_solid(self, solid):
        """Remove a solid from the solid list if it exists."""
        if solid in self.object['solid']:
            self.object['solid'].remove(solid)
