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


class Chem_Lab_Task_SL(BaseTask):
    #NOTE: we only cover here a subset of the task functions that are available,
    # checkout the base class for all the available functions to override.
    # ex: calculate_metrics, is_done..etc.
    def __init__(self, name):
        super().__init__(name=name, offset=None)
        self._goal_position = np.array([-0.3, -0.3, 0.0515 / 2.0])
        self._task_achieved = False
        self.Frankas = []
        self.Frankas_num = 7
        self._beaker_Kmno4_position = np.array([-2.757, -1.34, 0.1])
        self._beaker_Fecl2_position = np.array([-2.652, -1.34, 0.1])
        self._beaker_Fecl2_Return_position = np.array([-2.97, -1.13, 0.1])
        # self._beaker_Feo_position = np.array([-2.572, -1.362, 0.1])
        # self._Feo_position = self._beaker_Feo_position + np.array([0.0, 0.0, 0.02])
        self._Bottle_Kmno4_position = np.array([-2.063, -1.34, 0.1])
        self._Bottle_Fecl2_position = np.array([-2.16, -1.34, 0.1])
        self._pour0_offset = np.array([0.08, 0.00, 0.125])
        self._Fecl2_Bottle_Beaker_Pour_Offset = np.array([-0.078, 0.00, 0.125])
        self._pour1_offset = np.array([-0.06, 0.001, 0.091])

        return

    # Here we setup all the assets that we care about in this task.
    def set_up_scene(self, scene):
        super().set_up_scene(scene)

        self._Box_Liquid_Offset = Gf.Vec3f(-0.01, -0.02, 0.08)
        self._rng_seed = 42
        self._rng = np.random.default_rng(self._rng_seed)
        
        #object place setting

        Lab_path = '/home/huangyan/.local/share/ov/pkg/isaac_sim-2023.1.1/standalone_examples/Chem_lab/Controller_test.usd'
        Beaker_path = '/home/huangyan/.local/share/ov/pkg/isaac_sim-2023.1.1/standalone_examples/Chem_lab/Object_Usd/250_ml_beaker.usd'
        Bottle_Fecl2_path = Bottle_Kmno4_path = '/home/huangyan/.local/share/ov/pkg/isaac_sim-2023.1.1/standalone_examples/Chem_lab/Object_Usd/bottle_large1/bottle_large1.usd'
        # This will create a new XFormPrim and point it to the usd file as a reference
        # Similar to how pointers work in memory
        add_reference_to_stage(usd_path=Lab_path, prim_path="/World/Lab")    
        add_reference_to_stage(usd_path=Beaker_path, prim_path="/World/Lab/Beaker1")
        add_reference_to_stage(usd_path=Beaker_path, prim_path="/World/Lab/Beaker2")   
        # add_reference_to_stage(usd_path=Beaker_path, prim_path="/World/Lab/Beaker3")   
        add_reference_to_stage(usd_path=Bottle_Kmno4_path, prim_path="/World/Bottle_Kmno4")   
        add_reference_to_stage(usd_path=Bottle_Fecl2_path, prim_path="/World/Bottle_Fecl2")   
        self.Franka = scene.add(Franka(
                prim_path="/World/Lab/franka0",
                name=f"Franka",
                )
            )
        self._camera = scene.add(Camera(
                prim_path="/World/Lab/Camera",
                frequency = 30,
                resolution = [640,480],
                name=f"camera",
                )
            )        

        self._beaker_Kmno4 = scene.add(
            GeometryPrim(
                prim_path="/World/Lab/Beaker1",
                name=f"beaker_Kmno4",
                position = self._beaker_Kmno4_position,
                scale = np.array([0.8, 0.8, 0.88]),
                )
            )
        self._beaker_Fecl2 = scene.add(
            GeometryPrim(
                prim_path="/World/Lab/Beaker2",
                name=f"beaker_Fecl2",
                position = self._beaker_Fecl2_position,
                scale = np.array([0.8, 0.8, 0.7]),
                )
            )
        
        self._kmno4 = scene.add(
            GeometryPrim(
                prim_path="/World/Bottle_Kmno4",
                name=f"Bottle_Kmno4",
                position = self._Bottle_Kmno4_position,
                scale = np.array([0.8, 0.8, 0.9]),
                )
            )
        self._fecl2 = scene.add(
            GeometryPrim(
                prim_path="/World/Bottle_Fecl2",
                name=f"Bottle_Fecl2",
                position = self._Bottle_Fecl2_position,
                scale = np.array([0.8, 0.8, 0.9]),
                )
            )

        return

    # Information exposed to solve the task is returned from the task through get_observations
    def get_observations(self):
        # cube_position, _ = self._cube.get_world_pose()
        current_joint_positions = self.Franka.get_joint_positions()
        beaker_Kmno4_position, _ = self._beaker_Kmno4.get_world_pose()
        beaker_Fecl2_position, _ = self._beaker_Fecl2.get_world_pose()
        # beaker_Feo_position, _ = self._beaker_Feo.get_world_pose()
        kmno4_position, _ = self._kmno4.get_world_pose()
        kmno4_pour_position = np.add(beaker_Kmno4_position, self._pour0_offset)
        fecl2_position, _ = self._fecl2.get_world_pose()
        fecl2_pour_position = np.add(beaker_Fecl2_position, self._Fecl2_Bottle_Beaker_Pour_Offset)
        # feo_position, _ = self._feo.get_world_pose()
        beaker_Kmno4_pour_position = np.add(beaker_Fecl2_position, self._pour1_offset)
        # beaker_Fecl2_pour_position = np.add(beaker_Feo_position, self._pour1_offset)
        observations = {
            self.Franka.name: {
                "joint_positions": current_joint_positions,
            },
            self._beaker_Kmno4.name: {
                "Default_Position": self._beaker_Kmno4_position,
                "position": beaker_Kmno4_position,
                "Pour_Position":beaker_Kmno4_pour_position,
            },
            self._beaker_Fecl2.name: {
                "Default_Position": self._beaker_Fecl2_position,
                "position": beaker_Fecl2_position,
                # "Pour_Position":beaker_Fecl2_pour_position,
                "Return_Position":self._beaker_Fecl2_Return_position
            },
            # self._beaker_Feo.name: {
            #     "Default_Position": self._beaker_Feo_position,
            #     "position": beaker_Feo_position,
            # },
            self._kmno4.name: {
                "Default_Position": self._Bottle_Kmno4_position,
                "position": kmno4_position,
                "Pour_Position":kmno4_pour_position,
                "Return_Position": self._Bottle_Kmno4_position + + np.array([0.04, -0.02, 0]),
            },
            self._fecl2.name: {
                "Default_Position": self._Bottle_Fecl2_position,
                "position": fecl2_position,
                "Pour_Position":fecl2_pour_position,
            }
            # self._feo.name: {
            #     "Default_Position": self._Feo_position,
            #     "position": feo_position,
            # }
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
        # self.Franka.gripper.set_joint_positions(self.Franka.gripper.joint_opened_positions)
        # self._cube.get_applied_visual_material().set_color(color=np.array([0, 0, 1.0]))
        self._task_achieved = False
        return
    
    # def get_params(self) -> dict:
    #     """[summary]

    #     Returns:
    #         dict: [description]
    #     """
    #     params_representation = dict()
    #     params_representation["stack_target_position"] = {"value": self._stack_target_position, "modifiable": True}
    #     params_representation["franka_name"] = {"value": self._robot.name, "modifiable": False}
    #     return params_representation
