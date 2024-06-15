from omni.isaac.franka import Franka
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.stage import add_reference_to_stage
import numpy as np
from omni.isaac.core.prims import GeometryPrim
from omni.isaac.franka import Franka
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.tasks import BaseTask
from omni.isaac.sensor import Camera
from pxr import  Gf
import numpy as np
import os

# Get the current directory
current_directory = os.getcwd()

class Chem_Lab_Task(BaseTask):
    #NOTE: we only cover here a subset of the task functions that are available,
    # checkout the base class for all the available functions to override.
    # ex: calculate_metrics, is_done..etc.
    def __init__(self, name):
        super().__init__(name=name, offset=None)
        self._goal_position = np.array([-0.3, -0.3, 0.0515 / 2.0])
        self._task_achieved = False
        self._frankas = []
        self._frankas_num = 7
        self._Beaker1_position = np.array([-2.757, -1.34, 0.1])
        self._Beaker2_position = np.array([-2.652, -1.34, 0.1])
        self._Beaker2_Return_position = np.array([-2.97, -1.13, 0.1])
        # self._beaker_Feo_position = np.array([-2.572, -1.362, 0.1])
        # self._Feo_position = self._beaker_Feo_position + np.array([0.0, 0.0, 0.02])
        self._Bottle1_position = np.array([-2.063, -1.34, 0.1])
        self._Bottle2_position = np.array([-2.16, -1.34, 0.1])
        self._pour0_offset = np.array([0.08, 0.00, 0.125])
        self._Bottle2_Beaker_Pour_Offset = np.array([-0.078, 0.00, 0.125])
        self._pour1_offset = np.array([-0.06, 0.001, 0.091])

        return

    # Here we setup all the assets that we care about in this task.
    def set_up_scene(self, scene):
        super().set_up_scene(scene)

        self._Box_Liquid_Offset = Gf.Vec3f(-0.01, -0.02, 0.08)
        self._rng_seed = 42
        self._rng = np.random.default_rng(self._rng_seed)
        
        #object place setting

        Lab_path = os.path.join(current_directory,'Controller_test.usd')
        Beaker_path = os.path.join(current_directory,'Assets/beaker.usd')
        Bottle_Hcl_path = Bottle_Kmno4_path = os.path.join(current_directory,'Assets/bottle_large1/bottle_large1.usd')
        # This will create a new XFormPrim and point it to the usd file as a reference
        # Similar to how pointers work in memory
        add_reference_to_stage(usd_path=Lab_path, prim_path="/World/Lab")    
        add_reference_to_stage(usd_path=Beaker_path, prim_path="/World/Lab/Beaker1")
        add_reference_to_stage(usd_path=Beaker_path, prim_path="/World/Lab/Beaker2")   
        # add_reference_to_stage(usd_path=Beaker_path, prim_path="/World/Lab/Beaker3")   
        add_reference_to_stage(usd_path=Bottle_Kmno4_path, prim_path="/World/Bottle1")   
        add_reference_to_stage(usd_path=Bottle_Hcl_path, prim_path="/World/Bottle2")   
        self._franka = scene.add(Franka(
                prim_path="/World/Lab/franka0",
                name=f"Franka0",
                # usd_path = '/home/huangyan/isaac_sim-assets-1-2023.1.1/Assets/Isaac/2023.1.1/Isaac/Robots/Franka/franka.usd',
                )
            )
        self._camera = scene.add(Camera(
                prim_path="/World/Lab/Camera",
                frequency = 30,
                resolution = [640,480],
                name=f"camera",
                )
            )        
        # for Franka_idx in range(self._frankas_num):  # 从1到8
        #     franka_name = f"Franka{Franka_idx}"
        #     prim_path = f"/World/Room/{franka_name.lower()}"
        #     franka_instance = scene.add(Franka(
        #         prim_path=prim_path,
        #         name=franka_name,
        #     ))
        #     self._frankas.append(franka_instance)
        self._Beaker1 = scene.add(
            GeometryPrim(
                prim_path="/World/Lab/Beaker1",
                name=f"Beaker1",
                position = self._Beaker1_position,
                scale = np.array([0.8, 0.8, 0.88]),
                )
            )
        self._Beaker2 = scene.add(
            GeometryPrim(
                prim_path="/World/Lab/Beaker2",
                name=f"Beaker2",
                position = self._Beaker2_position,
                scale = np.array([0.8, 0.8, 0.7]),
                )
            )
        self._Bottle1 = scene.add(
            GeometryPrim(
                prim_path="/World/Bottle1",
                name=f"Bottle1",
                position = self._Bottle1_position,
                scale = np.array([0.8, 0.8, 0.9]),
                )
            )
        self._Bottle2 = scene.add(
            GeometryPrim(
                prim_path="/World/Bottle2",
                name=f"Bottle2",
                position = self._Bottle2_position,
                scale = np.array([0.8, 0.8, 0.9]),
                )
            )
        return

    # Information exposed to solve the task is returned from the task through get_observations
    def get_observations(self):
        current_joint_positions = self._franka.get_joint_positions()
        beaker1_position, _ = self._Beaker1.get_world_pose()
        beaker2_position, _ = self._Beaker2.get_world_pose()
        bottle1_position, _ = self._Bottle1.get_world_pose()
        bottle1_pour_position = np.add(beaker1_position, self._pour0_offset)
        bottle2_position, _ = self._Bottle2.get_world_pose()
        bottle2_pour_position = np.add(beaker2_position, self._Bottle2_Beaker_Pour_Offset)
        beaker1_pour_position = np.add(beaker2_position, self._pour1_offset)
        observations = {
            self._franka.name: {
                "joint_positions": current_joint_positions,
            },
            self._Beaker1.name: {
                "Default_Position": self._Beaker1_position,
                "position": beaker1_position,
                "Pour_Position":beaker1_pour_position,
                "Return_Position":self._Beaker1_position + np.array([0,0.02,0]),
                'Pour_Derection': -1,
            },
            self._Beaker2.name: {
                "Default_Position": self._Beaker2_position,
                "position": beaker2_position,
                # "Pour_Position":beaker2_pour_position,
                "Return_Position":self._Beaker2_Return_position
            },
            self._Bottle1.name: {
                "Default_Position": self._Bottle1_position,
                "position": bottle1_position,
                "Pour_Position":bottle1_pour_position,
                "Return_Position":self._Bottle1_position + np.array([0,0.02,0]),
                'Pour_Derection': -1,
            },
            self._Bottle2.name: {
                "Default_Position": self._Bottle2_position,
                "position": bottle2_position,
                "Pour_Position":bottle2_pour_position,
                "Return_Position":self._Bottle2_position + np.array([0,-0.02,0]),
                'Pour_Derection': 1,
            }
        }
        return observations



    # Called before each physics step,
    # for instance we can check here if the task was accomplished by
    # changing the color of the cube once its accomplished
    def pre_step(self, control_index, simulation_time):
        # # cube_position, _ = self._cube.get_world_pose()
        # if not self._task_achieved and np.mean(np.abs(self._goal_position - cube_position)) < 0.02:
        #     # Visual Materials are applied by default to the cube
        #     # in this case the cube has a visual material of type
        #     # PreviewSurface, we can set its color once the target is reached.
        #     self._cube.get_applied_visual_material().set_color(color=np.array([0, 1.0, 0]))
        #     self._task_achieved = True
        return

    # Called after each reset,
    # for instance we can always set the gripper to be opened at the beginning after each reset
    # also we can set the cube's color to be blue
    def post_reset(self):
        # self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)
        # self._cube.get_applied_visual_material().set_color(color=np.array([0, 0, 1.0]))
        self._task_achieved = False
        return
    
