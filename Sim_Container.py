#launch Isaac Sim before any other imports
#default first two lines in any standalone application
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from organic_example_task import Chem_Lab_Task_SL
from omni.isaac.franka import Franka
from omni.isaac.core.utils.types import ArticulationAction
from pxr import Sdf, Gf, UsdPhysics
from omni.isaac.sensor import Camera
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.physx.scripts import physicsUtils, particleUtils
from Controllers.Controller_Manager import ControllerManager
from Controllers.pick_move_controller import PickMoveController
from Controllers.pour_controller import PourController
from Controllers.return_controller import ReturnController
from Sim_Container import Sim_Container
from utils import Utils
from utils import *
import logging
import os
import matplotlib.pyplot as plt
from PIL import Image
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm

my_world = World(physics_dt = 1.0/ 120.0,stage_units_in_meters=1.0, set_defaults=False)
my_world._physics_context.enable_gpu_dynamics(flag=True)
stage = my_world.scene.stage
scenePath = Sdf.Path("/physicsScene")
utils = Utils()
utils._set_particle_parameter(my_world,particleContactOffset =  0.003)

my_world.add_task(Chem_Lab_Task_SL(name ='Chem_Lab_Task_SL'))
my_world.reset()

Franka0 = my_world.scene.get_object("Franka0")
mycamera = my_world.scene.get_object("camera")
current_observations = my_world.get_observations()
controller_manager = ControllerManager(my_world, Franka0, Franka0.gripper)
# particle params#4*5 is 1ml

Sim_Bottle_Kmno4 = Sim_Container(
                                world = my_world,     
                                sim_container = my_world.scene.get_object("Bottle_Kmno4"),
                                solute={'c1ccc2cc3ccccc3cc2c1': 10}, 
                                org=True,  
                                volume=10
                                )
Sim_Bottle_Hcl = Sim_Container(
                                world = my_world, 
                                sim_container = my_world.scene.get_object("Bottle_Hcl"),
                                solute={'BrBr': 20}, 
                                org=True,  
                                volume=10
                                )
Sim_Beaker_Kmno4 = Sim_Container(my_world,sim_container = my_world.scene.get_object("beaker_Kmno4"),org=True)
Sim_Beaker_Hcl = Sim_Container(my_world,sim_container = my_world.scene.get_object("beaker_Hcl"),org=True)

Sim_Beaker_Kmno4.sim_update(Sim_Bottle_Kmno4,Franka0,controller_manager)
Sim_Beaker_Hcl.sim_update(Sim_Bottle_Hcl,Franka0,controller_manager,5)
Sim_Beaker_Hcl.sim_update(Sim_Beaker_Kmno4,Franka0,controller_manager)

count = 1
root_path = '/home/huangyan/.local/share/ov/pkg/isaac_sim-2022.2.1/standalone_examples/Chem_lab/Organic_demo'

while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()
            controller_manager.reset()
        current_observations = my_world.get_observations()
        controller_manager.execute(current_observations = my_world.get_observations())
        controller_manager.process_concentration_iters()
        if controller_manager.need_new_liquid():
            controller_manager.get_current_controller()._get_sim_container2().create_liquid(controller_manager,current_observations)
        if (count%10 == 0):
            img = mycamera.get_rgba()
            file_name = os.path.join(root_path, f"{count}")
            save_rgb(img, file_name)
        count += 1 
        if controller_manager.is_done():
            image_files = [os.path.join(root_path, f) for f in os.listdir(root_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            image_files.sort(key=natural_sort_key)  # 使用自然排序
            clip = ImageSequenceClip(image_files, fps=60)
            clip.write_videofile(os.path.join(root_path, "output_video.mp4"), codec="libx264")
            for image_file in tqdm(image_files):
                os.remove(image_file)
            my_world.pause()
            break
simulation_app.close()