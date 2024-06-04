from LLM.agent import AgentLLM as Agent
from utils import *
import functools
from Controllers.Controller_Manager import ControllerManager
from Controllers.pick_move_controller import PickMoveController
from Controllers.pour_controller import PourController
from Controllers.return_controller import ReturnController
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from chem_sim.simulation.database import reactions

import numpy as np
from pxr import Sdf, Gf, UsdPhysics
from Sim_Container import Sim_Container

PROMPTS_PATH = '/home/huangyan/.local/share/ov/pkg/isaac_sim-2023.1.1/standalone_examples/Chem_lab/prompts_JSON'
LOG_PATH = '/home/huangyan/.local/share/ov/pkg/isaac_sim-2023.1.1/standalone_examples/Chem_lab/LLM/log'

class MAS:
    """
    Multi-Agent System (MAS) for managing and simulating chemical reactions and robotics control in a simulated environment.
    
    Args:
        world: The simulation world object.
        controller_manager: The controller manager object.
    """
    
    def __init__(self, world, controller_manager) -> None:
        self.my_world = world
        self._observation = self.my_world.get_observations()
        
        self.plan_steps_list = []
        self.plan_message_str = ''  # Used for debugging
        self.code_str = ''
        self.reaction_dict = reactions
        self.generated_func_str = ''
        
        self.max_num_retry = 3
        self.utils = Utils(self.my_world)
        self.initial_coder_function_dict = {
            'scenePath': Sdf.Path("/physicsScene"),
            'Gf': Gf,
            'UsdPhysics': UsdPhysics,
            'my_world': self.my_world,
            'current_observations': self.my_world.get_observations(),
            'Franka': self.my_world.scene.get_object("Franka"),
            'Bottle_Kmno4': self.my_world.scene.get_object("Bottle_Kmno4"),
            'beaker_Kmno4': self.my_world.scene.get_object("beaker_Kmno4"),
            'Bottle_Fecl2': self.my_world.scene.get_object("Bottle_Fecl2"),
            'beaker_Fecl2': self.my_world.scene.get_object("beaker_Fecl2"),
            'Feo': self.my_world.scene.get_object("Feo"),
            'Sim_Container': Sim_Container,
            'controller_manager': controller_manager,
            'PickMoveController': PickMoveController,
            'PourController': PourController,
            'ReturnController': ReturnController,
            'RMPFlowController': RMPFlowController,
            'euler_angles_to_quat': euler_angles_to_quat,
            'np': np,
            'utils': self.utils,
        }
        self.coder_function_dict = self.initial_coder_function_dict.copy()
        
        self.planner_prompt_filename = '/planner_prompt.txt'
        self.agents_initialization()
    
    def agents_initialization(self):
        """
        Initialize agents for different tasks and load their system prompts.
        """
        self.agent_controller_generator = Agent("controller_generator", save_path=LOG_PATH)
        self.agent_planner = Agent("planner", save_path=LOG_PATH)
        self.agent_coder = Agent("coder", save_path=LOG_PATH)
        self.agent_debugger = Agent("debugger", save_path=LOG_PATH)
        self.agent_reaction_responser = Agent("reaction_responser", save_path=LOG_PATH)
        self.agent_add_controllers = Agent("add_controllers", save_path=LOG_PATH)
        self.agent_add_sim_containers = Agent("add_sim_containers", save_path=LOG_PATH)
        self.agent_add_rigidbody = Agent("add_rigidbody", save_path=LOG_PATH)
        self.agent_add_particles = Agent("add_particles", save_path=LOG_PATH)
        self.agent_add_tasks = Agent("add_tasks", save_path=LOG_PATH)
        
        # Load system prompts
        self.agent_controller_generator.load_system_prompt_from_file(PROMPTS_PATH + '/controller_generator_prompt.txt')
        self.agent_reaction_responser.load_system_prompt_from_file(PROMPTS_PATH + '/reaction_responser_prompt.txt')
        self.agent_add_controllers.load_system_prompt_from_file(PROMPTS_PATH + '/add_controller_prompt.txt')
        self.agent_add_sim_containers.load_system_prompt_from_file(PROMPTS_PATH + '/add_sim_container_prompt.txt')
        self.agent_add_rigidbody.load_system_prompt_from_file(PROMPTS_PATH + '/add_rigid_body_prompt.txt')
        self.agent_add_particles.load_system_prompt_from_file(PROMPTS_PATH + '/add_particle_set_prompt.txt')
        self.agent_add_tasks.load_system_prompt_from_file(PROMPTS_PATH + '/add_tasks_prompt.txt')
        self.agent_coder.load_system_prompt_from_file(PROMPTS_PATH + '/coder_prompt.txt')
        self.agent_debugger.load_system_prompt_from_file(PROMPTS_PATH + '/debugger_prompt.txt')
        
    def _update_system_prompts(self):
        """
        Update system prompts for all agents.
        """
        self.agent_add_controllers.load_system_prompt_from_file(PROMPTS_PATH + '/controller_generator_prompt.txt')
        self.agent_reaction_responser.load_system_prompt_from_file(PROMPTS_PATH + '/reaction_responser_prompt.txt')
        self.agent_add_sim_containers.load_system_prompt_from_file(PROMPTS_PATH + '/add_sim_container_prompt.txt')
        self.agent_add_rigidbody.load_system_prompt_from_file(PROMPTS_PATH + '/add_rigid_body_prompt.txt')
        self.agent_add_particles.load_system_prompt_from_file(PROMPTS_PATH + '/add_particle_set_prompt.txt')
        self.agent_add_tasks.load_system_prompt_from_file(PROMPTS_PATH + '/add_tasks_prompt.txt')
        self.agent_coder.load_system_prompt_from_file(PROMPTS_PATH + '/coder_prompt.txt')
        self.agent_debugger.load_system_prompt_from_file(PROMPTS_PATH + '/debugger_prompt.txt')
        
    def _generate_plan(self, controllers_str):
        """
        Generate a plan for a given task.

        Args:
            controllers_str (str): The input string describing the controllers.

        Returns:
            None
        """
        user_prompt = controllers_str
        message = self.agent_planner.generate_response(user_prompt)
        self.plan_steps_list = extract_scripts(message)
        print(f'Number of generated plan steps: {len(self.plan_steps_list)}')
    
    def _response_reaction(self, expected_chem):
        """
        Respond to a chemical reaction.

        Args:
            expected_chem (str): The expected chemical reaction.

        Returns:
            str: The response message.
        """
        observation = "observation: " + str(self._observation.keys()) + '\\n'
        user_prompt = observation + 'reaction_dict:' + str(self.reaction_dict) + '\\n' + expected_chem
        message = self.agent_reaction_responser.generate_response(user_prompt)
        return message
    
    def _add_particles(self):
        """
        Add particles for each object.

        Returns:
            str: The response message.
        """
        observation = str(self._observation)
        message = self.agent_add_particles.generate_response(observation)
        return message
    
    def _add_sim_container(self, particle_set_str):
        """
        Add simulation containers for each object.

        Args:
            particle_set_str (str): The string describing the particle set.

        Returns:
            str: The response message.
        """
        observation = "observation: " + str(self._observation) + '\\n'
        user_prompt = observation + particle_set_str
        message = self.agent_add_sim_containers.generate_response(user_prompt)
        return message
    
    def _add_rigidbody(self):
        """
        Add rigidbody for each object.

        Returns:
            str: The response message.
        """
        added_objects_dict = self.get_added_coder_function_dict()
        added_objects_dict_str = 'Objects introduced in the scene: ' + str(added_objects_dict.keys()) + '\\n'
        message = self.agent_add_rigidbody.generate_response(added_objects_dict_str)
        return message
    
    def _add_controllers(self, controllers_str):
        """
        Add controllers for a given task.

        Args:
            controllers_str (str): The input string describing the controllers.

        Returns:
            str: The response message.
        """
        user_prompt = controllers_str
        message = self.agent_add_controllers.generate_response(user_prompt)
        return message
    
    def _add_tasks(self, controllers_str):
        """
        Add tasks for a given controller.

        Args:
            controllers_str (str): The input string describing the controllers.

        Returns:
            str: The response message.
        """
        user_prompt = controllers_str
        message = self.agent_add_tasks.generate_response(user_prompt)
        return message
    
    def _generate_controllers(self, prompt, observation):
        """
        Generate controllers for a given task.

        Args:
            prompt (str): The task description.
            observation (dict): The current observations.

        Returns:
            str: The response message.
        """
        self.observation_str = self._observations_to_string(observation)
        instantiated_objects = f"'Instantiated objects: '{self.coder_function_dict}\\n"
        total_prompt = f"'observation: '{self.observation_str}\n{prompt}"
        
        message = self.agent_controller_generator.generate_response(total_prompt)
        return message
    
    def _observations_to_string(self, observation):
        """
        Convert observation dictionary into a descriptive string.

        Args:
            observation (dict): The current observations.

        Returns:
            str: The observation
