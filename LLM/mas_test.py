from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from mas_task import Chem_Lab_Task_SL
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
import logging
import os
import matplotlib.pyplot as plt
from PIL import Image
import time
from mas import *
import threading

my_world = World(physics_dt = 1.0/ 120.0,stage_units_in_meters=1.0, set_defaults=False)
my_world._physics_context.set_broadphase_type('GPU')
my_world._physics_context.enable_gpu_dynamics(flag=True)
utils = Utils(my_world)
stage = my_world.scene.stage
scenePath = Sdf.Path("/physicsScene")
task = Chem_Lab_Task_SL(name ='Chem_Lab_Task_SL')
my_world.add_task(task)
my_world.reset()

Franka = my_world.scene.get_object("Franka")

Bottle_Kmno4 = my_world.scene.get_object("Bottle_Kmno4")
Bottle_Fecl2 = my_world.scene.get_object("Bottle_Fecl2")
beaker_Kmno4 = my_world.scene.get_object("beaker_Kmno4")
beaker_Fecl2 = my_world.scene.get_object("beaker_Fecl2")


my_dict = {
    'Franka': my_world.scene.get_object("Franka"),
    'Bottle_Kmno4': my_world.scene.get_object("Bottle_Kmno4"),
    'beaker_Kmno4': my_world.scene.get_object("beaker_Kmno4"),
    'Bottle_Fecl2': my_world.scene.get_object("Bottle_Fecl2"),
    'beaker_Fecl2': my_world.scene.get_object("beaker_Fecl2")
}

current_observations = my_world.get_observations()
# particle params#4*5 is 1ml
utils._set_particle_parameter(particleContactOffset =  0.003)
# initialize the agent system
controller_manager = ControllerManager(my_world, Franka)
mas = MAS(my_world,controller_manager)

add_particles_str = mas._add_particles()
with open('add_particle_set_str.txt', 'w') as file:
    file.write(add_particles_str)

mas._generate_code_str(add_particles_str)
mas._execute_code_str()

sim_container_str = mas._add_sim_container(add_particles_str)
with open('add_sim_container_str.txt', 'w') as file:
    file.write(sim_container_str)

mas._generate_code_str(sim_container_str)
mas._execute_code_str()

add_rigidbody_str = mas._add_rigidbody()
with open('add_rigidbody_str.txt', 'w') as file:
    file.write(add_rigidbody_str)
mas._generate_code_str(add_rigidbody_str)
mas._execute_code_str()

user = 'Please observe the test bench and tell me what kind of chemical experiments I can do'
print(mas._response_reaction(user))


user_prompt = None
controllers_ready = False

def get_user_input():
    global user_prompt
    while True:
        user_prompt = input("Enter your task: ")

# Start the input thread
input_thread = threading.Thread(target=get_user_input)
input_thread.daemon = True
input_thread.start()

while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()
            controller_manager.reset()
        
        current_observations = my_world.get_observations()
        
        if user_prompt and not controllers_ready:
            
            print('code generating ...')
            # Generate controllers based on the user prompt
            controllers_str = mas._generate_controllers(user_prompt, current_observations)
            with open('controllers_str.txt', 'w') as file:
                file.write(controllers_str)
            with open('controllers_str.txt', 'r') as f:
                controllers_str = f.read()
            mas._generate_code_str(controllers_str)
            mas._execute_code_str()

            add_controllers_str = mas._add_controllers(controllers_str)
            with open('add_controllers_str.txt', 'w') as file:
                file.write(add_controllers_str)
            mas._generate_code_str(add_controllers_str)
            mas._execute_code_str()

            add_tasks_str = mas._add_tasks(add_controllers_str + str(my_dict))
            with open('add_tasks_str.txt', 'w') as file:
                file.write(add_tasks_str)
            mas._generate_code_str(add_tasks_str)
            mas._execute_code_str()

            controllers_ready = True
            print('controllers executing ...')
        
        if controllers_ready:
            # Execute the controller manager
            controller_manager.execute(current_observations=current_observations)

            if controller_manager.is_done():
                my_world.pause()

                controllers_ready = False  # Reset for next user prompt
                user_prompt = None  



simulation_app.close()
