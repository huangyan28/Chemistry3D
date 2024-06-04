from utils import Utils
from pxr import Sdf, Gf, UsdPhysics

class ControllerManager:
    """
    A manager class for handling multiple controllers and their tasks in a robotic simulation environment.

    Args:
        world: The simulation world instance.
        franka: The robotic arm instance.
        gripper: The gripper instance.
    """

    def __init__(self, world, franka, gripper):
        self.world = world
        self.franka = franka
        self.gripper = gripper
        self.controllers = []  # List to store controller instances
        self.tasks = []  # List to store tasks
        self.current_task_index = 0  # Index of the current task
        self.concentration_iters = []  # List to store concentration iterators
        self._need_new_liquid = False  # Flag to indicate if new liquid is needed
        self.new_liquid_color = None  # Color of the new liquid
        self._need_solid_melt = False  # Flag to indicate if solid needs to be melted

    def add_controller(self, controller_name, controller_instance):
        """
        Add a controller instance to the list of controllers.

        Args:
            controller_name (str): The name of the controller.
            controller_instance: The instance of the controller to add.
        """
        if controller_name in [c["controller_name"] for c in self.controllers]:
            # If a controller with the same name exists, add an index to ensure uniqueness
            index = 1
            while f"{controller_name}{index}" in [c["controller_name"] for c in self.controllers]:
                index += 1
            controller_name = f"{controller_name}{index}"
        self.controllers.append({"controller_name": controller_name, "controller_instance": controller_instance})

    def add_task(self, controller_type, param_template):
        """
        Add a task to the list of tasks.

        Args:
            controller_type (str): The type of the controller for the task.
            param_template (dict): A template for the task parameters.
        """
        self.tasks.append({"controller_type": controller_type, "param_template": param_template})

    def get_current_controller_name(self):
        """
        Get the name of the current controller.

        Returns:
            str: The name of the current controller.
        """
        return self.controllers[self.current_task_index]['controller_name']

    def get_current_controller(self):
        """
        Get the current controller instance.

        Returns:
            The current controller instance.
        """
        return self.controllers[self.current_task_index]['controller_instance']

    def execute(self, current_observations):
        """
        Execute the current task using the current controller.

        Args:
            current_observations (dict): The current observations from the simulation.
        """
        if self.is_done():
            print("All tasks completed")
            self.world.pause()
            return
        
        task = self.tasks[self.current_task_index]
        controller = self.get_current_controller()

        controller_name = self.get_current_controller_name()
        controller_type = task["controller_type"]
        param_template = task["param_template"]

        # Ensure the controller exists
        if controller_name not in [c["controller_name"] for c in self.controllers]:
            print(f"Controller {controller_name} not found!")
            return
        
        # Generate actual parameters based on param_template
        task_params = self.generate_task_params(param_template, current_observations)

        actions = controller.forward(**task_params)
        if controller_type == 'pour':
            if controller.check_need_for_new_liquid() and controller.started_pouring():
                controller.liquid_created_set(True)
                self.set_need_new_liquid(True)
                self.new_liquid_color = controller.get_new_liquid_color()

            if controller.reaction_started():
                if not controller.iter_added():
                    self.concentration_iters.append(controller.get_concentrations_iter())
                    controller.set_iter_added(True)
                controller.update_sim_container_color()

            if controller.check_need_solid_melt() and controller.reaction_started():
                controller.solid_melted_set(True)
                self.set_need_solid_melt(True)

        self.franka.apply_action(actions)
        if controller.is_done():
            print(f"{controller_name} is done")
            self.current_task_index += 1  # Move to the next task

    def generate_task_params(self, param_template, current_observations):
        """
        Generate task parameters based on a template and current observations.

        Args:
            param_template (dict): The template for task parameters.
            current_observations (dict): The current observations from the simulation.

        Returns:
            dict: The generated task parameters.
        """
        task_params = {}
        for param, value in param_template.items():
            if callable(value):
                task_params[param] = value(current_observations)
            else:
                task_params[param] = value
        return task_params

    def get_value_by_path(self, observations, path):
        """
        Get a value from nested dictionaries based on a path.

        Args:
            observations (dict): The observations dictionary.
            path (str): The path to the value.

        Returns:
            The value at the specified path.
        """
        # Logic to get value from nested dictionary if needed
        return observations[path]

    def process_concentration_iters(self):
        """
        Process concentration iterators to print their next values.

        Remove iterators that have been exhausted.
        """
        to_remove = []
        for iter_index, conc_iter in enumerate(self.concentration_iters):
            try:
                print(next(conc_iter))
            except StopIteration:
                to_remove.append(iter_index)
        
        for index in reversed(to_remove):
            self.concentration_iters.pop(index)

    def need_solid_melt(self):
        """
        Check if solid needs to be melted.

        Returns:
            bool: True if solid needs to be melted, False otherwise.
        """
        return self._need_solid_melt

    def set_need_solid_melt(self, need_solid_melt):
        """
        Set the flag indicating if solid needs to be melted.

        Args:
            need_solid_melt (bool): The flag to set.
        """
        self._need_solid_melt = need_solid_melt

    def set_need_new_liquid(self, need_new_liquid):
        """
        Set the flag indicating if new liquid needs to be created in Sim_Container2.

        Args:
            need_new_liquid (bool): The flag to set.
        """
        self._need_new_liquid = need_new_liquid

    def need_new_liquid(self):
        """
        Get the status of whether new liquid needs to be created.

        Returns:
            bool: True if new liquid needs to be created, False otherwise.
        """
        return self._need_new_liquid

    def get_current_liquid_color(self):
        """
        Get the color of the new liquid from the current controller.

        Returns:
            The color of the new liquid.
        """
        return self.get_current_controller().get_new_liquid_color()

    def reset(self):
        """
        Reset the task index and all controllers.
        """
        self.current_task_index = 0
        for controller in [c["controller_instance"] for c in self.controllers]:
            controller.reset()

    def is_done(self) -> bool:
        """
        Check if all tasks have been completed.

        Returns:
            bool: True if all tasks are completed, False otherwise.
        """
        return self.current_task_index >= len(self.tasks)
