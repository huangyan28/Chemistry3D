from omni.physx.scripts import particleUtils
from pxr import Gf, Vt, UsdPhysics, PhysxSchema, UsdShade, Sdf
import omni.timeline
import numpy as np
import omni.kit.commands
from chem_sim.simulation.simulator import Container
import re, json
import pickle, os
import ast

# utils.py

class Utils:
    def __init__(self, hello_world_instance):
        """
        Initialize the Utils class with the simulation world instance and set random number generator seed.
        Args:
            hello_world_instance: The simulation world instance.
        """
        self._world = hello_world_instance
        self._rng_seed = 42
        self._rng = np.random.default_rng(self._rng_seed)
        self._set_particle_parameter()

    def _set_particle_parameter(self, particleContactOffset=0.003):
        """
        Set parameters for particle simulation.
        Args:
            particleContactOffset (float): The contact offset for particles.
        """
        self._world._ContactOffset = particleContactOffset
        self._world._particleContactOffset = particleContactOffset
        self._world._restOffset = particleContactOffset * 0.99
        self._world._Solid_Rest_Offset = particleContactOffset * 0.9
        self._world._fluidRestOffset = self._world._restOffset * 0.6
        self._world._particleSpacing = 2 * self._world._fluidRestOffset
        return

    def _add_particle_system(self, particle_system_path, simulation_owner):
        """
        Add a PhysX particle system to the simulation world.
        Args:
            particle_system_path (str): Path to the particle system.
            simulation_owner: The owner of the simulation.
        """
        if not hasattr(self._world, '_solverPositionIterations'):
            self._world._solverPositionIterations = 16

        particle_system = particleUtils.add_physx_particle_system(
            stage=self._world.scene.stage,
            particle_system_path=particle_system_path,
            contact_offset=self._world._ContactOffset,
            rest_offset=self._world._restOffset * 1.5,
            particle_contact_offset=self._world._particleContactOffset,
            solid_rest_offset=self._world._Solid_Rest_Offset,
            fluid_rest_offset=self._world._fluidRestOffset,
            solver_position_iterations=self._world._solverPositionIterations,
            simulation_owner=simulation_owner
        )
        particle_system.CreateMaxVelocityAttr().Set(self._world._restOffset * 1000.0)

        return particle_system

    def _set_isosurface_particle_system(self, particlesystem):
        """
        Set isosurface parameters for a given particle system.
        Args:
            particlesystem: The particle system to set parameters for.
        """
        isosurfaceAPI = PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(particlesystem.GetPrim())
        isosurfaceAPI.CreateIsosurfaceEnabledAttr().Set(True)
        isosurfaceAPI.CreateMaxVerticesAttr().Set(1024 * 1024)
        isosurfaceAPI.CreateMaxTrianglesAttr().Set(2 * 1024 * 1024)
        isosurfaceAPI.CreateMaxSubgridsAttr().Set(1024 * 2)
        isosurfaceAPI.CreateGridSpacingAttr().Set(self._world._fluidRestOffset * 1.2)
        isosurfaceAPI.CreateSurfaceDistanceAttr().Set(self._world._fluidRestOffset * 1.6)
        isosurfaceAPI.CreateGridFilteringPassesAttr().Set("")
        isosurfaceAPI.CreateGridSmoothingRadiusAttr().Set(self._world._fluidRestOffset * 2)
        isosurfaceAPI.CreateGridFilteringPassesAttr().Set('GSRS')
        isosurfaceAPI.CreateNumMeshSmoothingPassesAttr().Set(4)
        isosurfaceAPI.CreateNumMeshNormalSmoothingPassesAttr().Set(4)

    def _add_particle_set(self, particle_set_path, simulation_owner, particle_system_path, dim_x=12, dim_y=12, dim_z=12, center=None):
        """
        Add a particle set to the simulation world.
        Args:
            particle_set_path (str): Path to the particle set.
            simulation_owner: The owner of the simulation.
            particle_system_path (str): Path to the particle system.
            dim_x (int): Number of particles in the x-dimension.
            dim_y (int): Number of particles in the y-dimension.
            dim_z (int): Number of particles in the z-dimension.
            center (list): Center coordinates for the particle set.
        """
        if center is None:
            center = self._world._Box_Liquid_Offset

        half_dim_x = dim_x * 0.5
        half_dim_y = dim_y * 0.5
        half_dim_z = dim_z * 0.5

        start_x = center[0] - half_dim_x * self._world._particleSpacing
        start_y = center[1] - half_dim_y * self._world._particleSpacing
        start_z = center[2]

        positions, velocities = particleUtils.create_particles_grid(
            Gf.Vec3f(start_x, start_y, start_z),
            self._world._particleSpacing * 1.1,
            dim_x,
            dim_y,
            dim_z,
        )
        widths = [self._world._particleSpacing] * len(positions)
        particle_set = particleUtils.add_physx_particleset_points(
            stage=self._world.scene.stage,
            path=particle_set_path,
            positions_list=Vt.Vec3fArray(positions),
            velocities_list=Vt.Vec3fArray(velocities),
            widths_list=widths,
            particle_system_path=particle_system_path,
            self_collision=True,
            fluid=True,
            particle_group=0,
            particle_mass=0.0,
            density=1.0,
        )

        return particle_set

    def merge_particle_sets(self, stage, merged_particle_set_path, particle_set1_path, particle_set2_path):
        """
        Merge two particle sets into a new particle set.
        Args:
            stage: The USD stage.
            merged_particle_set_path (str): Path for the merged particle set.
            particle_set1_path (str): Path for the first particle set.
            particle_set2_path (str): Path for the second particle set.
        """
        merged_particle_set = UsdPhysics.ParticleSet.Define(stage, merged_particle_set_path)

        positions1 = stage.GetAttribute(particle_set1_path.AppendProperty("positions")).Get()
        velocities1 = stage.GetAttribute(particle_set1_path.AppendProperty("velocities")).Get()
        widths1 = stage.GetAttribute(particle_set1_path.AppendProperty("widths")).Get()

        positions2 = stage.GetAttribute(particle_set2_path.AppendProperty("positions")).Get()
        velocities2 = stage.GetAttribute(particle_set2_path.AppendProperty("velocities")).Get()
        widths2 = stage.GetAttribute(particle_set2_path.AppendProperty("widths")).Get()

        merged_positions = Vt.Vec3fArray(positions1 + positions2)
        merged_velocities = Vt.Vec3fArray(velocities1 + velocities2)
        merged_widths = Vt.FloatArray(widths1 + widths2)

        merged_particle_set.CreatePositionsAttr(merged_positions)
        merged_particle_set.CreateVelocitiesAttr(merged_velocities)
        merged_particle_set.CreateWidthsAttr(merged_widths)

        return merged_particle_set

    def create_and_bind_mdl_material(self, mdl_name, mtl_name, prim_path):
        """
        Create and bind an MDL material to a specified prim path.
        Args:
            mdl_name (str): Name of the MDL material.
            mtl_name (str): Name of the material.
            prim_path (str): Path of the prim to bind the material to.
        """
        mtl_created_list = []
        omni.kit.commands.execute(
            "CreateAndBindMdlMaterialFromLibrary",
            mdl_name=mdl_name,
            mtl_name=mtl_name,
            mtl_created_list=mtl_created_list,
        )

        material_shader = UsdShade.Shader.Get(self._world.scene.stage, mtl_created_list[0] + '/Shader')

        omni.kit.commands.execute(
            "BindMaterial", prim_path=prim_path, material_path=mtl_created_list[0]
        )

        return mtl_created_list[0], material_shader

    def create_particle_system_and_set(self, particle_system_path_str, particle_set_path_str, scenePath, center, dim_x, dim_y, dim_z, material_color, mdl_name="OmniSurfacePresets.mdl", mtl_name="OmniSurface_ClearWater"):
        """
        Create a particle system and particle set with specified parameters and bind a material.
        Args:
            particle_system_path_str (str): Path for the particle system.
            particle_set_path_str (str): Path for the particle set.
            scenePath (str): Path of the scene.
            center (list): Center coordinates for the particle set.
            dim_x (int): Number of particles in the x-dimension.
            dim_y (int): Number of particles in the y-dimension.
            dim_z (int): Number of particles in the z-dimension.
            material_color (list): RGB color values for the material.
            mdl_name (str): Name of the MDL material library.
            mtl_name (str): Name of the material.
        """
        particle_system_path = Sdf.Path(particle_system_path_str)
        particle_set_path = Sdf.Path(particle_set_path_str)

        particle_system = self._add_particle_system(
            particle_system_path=particle_system_path,
            simulation_owner=scenePath,
        )

        _, material_shader = self.create_and_bind_mdl_material(
            mdl_name=mdl_name,
            mtl_name=mtl_name,
            prim_path=particle_system_path,
        )

        if not isinstance(material_color, Gf.Vec3f):
            if isinstance(material_color, (list, tuple)) and len(material_color) == 3:
                material_color = Gf.Vec3f(*material_color)
            else:
                raise ValueError("material_color must be a list or tuple of three floats, or a Gf.Vec3f instance")

        material_shader.CreateInput("specular_transmission_color", Sdf.ValueTypeNames.Color3f).Set(material_color)

        particle_set = self._add_particle_set(
            particle_set_path=particle_set_path,
            simulation_owner=scenePath,
            particle_system_path=particle_system_path,
            dim_x=dim_x,
            dim_y=dim_y,
            dim_z=dim_z,
            center=center
        )

        self._set_isosurface_particle_system(particle_system)

        particle_system_set_material_dict = {
            'particle_system': particle_system,
            'particle_set': particle_set,
            'material_shader': material_shader
        }

        return particle_system_set_material_dict

def material_color_set(material_shader, material_color):
    """
    Set the color of a material shader.
    Args:
        material_shader: The material shader to set the color for.
        material_color (list): RGB color values.
    """
    material_shader.CreateInput("specular_transmission_color", Sdf.ValueTypeNames.Color3f).Set(material_color)
    return

def get_ParticleSet_Centroid(particle_set):
    """
    Calculate the centroid of a particle set.
    Args:
        particle_set: The particle set to calculate the centroid for.
    """
    particle_position = particle_set.GetPointsAttr().Get()
    points_array = np.array(particle_position)
    centroid = np.median(points_array, axis=0)
    return centroid

def is_contact(centroid1, centroid2, threshold=0.05):
    """
    Determine if two centroids are within a specified distance threshold.
    Args:
        centroid1 (list): The first centroid coordinates.
        centroid2 (list): The second centroid coordinates.
        threshold (float): The distance threshold.
    """
    centroid1_array = np.array(centroid1)
    centroid2_array = np.array(centroid2)
    distance = np.linalg.norm(centroid1_array - centroid2_array)
    return distance < threshold

def transfer_json_string_to_dict(json_string):
    """
    Convert a JSON string to a dictionary.
    Args:
        json_string (str): A JSON string.
    Returns:
        dict: A dictionary representation of the JSON string.
    """
    if isinstance(json_string, dict):
        return json_string

    json_string = json_string.strip()
    scripts = extract_scripts(json_string)
    if len(scripts) > 0:
        json_string = scripts[0]
    try:
        output_dict = json.loads(json_string)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    return output_dict

def exec_code(code: str):
    """
    Execute a given string of Python code.
    Args:
        code (str): The Python code to execute.
    """
    try:
        exec(code)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def extract_scripts(text, pattern=None):
    """
    Extract scripts enclosed within triple double quotes from the given text.
    Args:
        text (str): The text containing the scripts.
        pattern (str): The regex pattern to use for extraction.
    Returns:
        list: A list of extracted scripts.
    """
    if pattern is None:
        pattern = r'```(.*?)```'

    scripts = re.findall(pattern, text, re.DOTALL)

    for i in range(len(scripts)):
        scripts[i] = '\n'.join(scripts[i].split('\n')[1:])

    return scripts

def combine_scripts(scripts):
    """
    Combine a list of scripts into a single script.
    Args:
        scripts (list): A list of script strings.
    Returns:
        str: A single combined script.
    """
    combined_script = '\n'.join(scripts)
    return combined_script

def cover_file(source_path, destination_path):
    """
    Overwrite the contents of a destination file with the contents of a source file.
    Args:
        source_path (str): Path to the source file.
        destination_path (str): Path to the destination file.
    """
    try:
        with open(source_path, 'r') as source_file, open(destination_path, 'w') as destination_file:
            content = source_file.read()
            destination_file.write(content)
        print(f"The file {destination_path} has been successfully covered by the contents of {source_path}.")
    except FileNotFoundError:
        print(f"One of the files at {source_path} or {destination_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def save_mas_instance(mas_instance, filename):
    """
    Save a Multi-Agent System (MAS) instance to a file.
    Args:
        mas_instance: The MAS instance to save.
        filename (str): The filename to save the instance to.
    """
    with open(filename, 'wb') as f:
        pickle.dump(mas_instance, f)

def save_as_json(data, filename):
    """
    Save data as a JSON file.
    Args:
        data: The data to save.
        filename (str): The filename to save the data to.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(f'{filename}', 'w') as f:
        json.dump(data, f)

def load_from_json(filename):
    """
    Load data from a JSON file.
    Args:
        filename (str): The filename to load data from.
    Returns:
        The loaded data.
    """
    if not os.path.exists(filename):
        print(f"load_from_json: File {filename} does not exist.")
        return None
    with open(f'{filename}', 'r') as f:
        data = json.load(f)
    return data

def transfer_dict_to_json_string(input_dict):
    """
    Convert a dictionary to a JSON string.
    Args:
        input_dict (dict): The dictionary to convert.
    Returns:
        str: A JSON string representation of the dictionary.
    """
    try:
        json_string = json.dumps(input_dict, indent=4)
        return json_string
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def get_function_name(code):
    """
    Get the name of the function defined in a given code string.
    Args:
        code (str): The code containing the function.
    Returns:
        str: The name of the function.
    """
    if len(extract_scripts(code)) > 0:
        code = extract_scripts(code)[0]
    module = ast.parse(code)
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            return node.name

def split_script_by_function_with_args(script, function_name='observe'):
    """
    Split a Python script into segments based on the occurrence of a specific function with any arguments.
    Args:
        script (str): The Python script as a string.
        function_name (str): The name of the function to split the script by.
    Returns:
        list: A list of script segments.
    """
    pattern = r'{}\(.*?\)'.format(re.escape(function_name))

    segments = []
    current_segment = []
    for line in script.split('\n'):
        if line.strip().startswith('#'):
            current_segment.append(line)
            continue

        matches = list(re.finditer(pattern, line))
        if matches:
            start = 0
            for match in matches:
                current_segment.append(line[start:match.start()])
                segments.append('\n'.join(current_segment))
                current_segment = [match.group()]
                start = match.end()

            current_segment.append(line[start:])
        else:
            current_segment.append(line)

    if current_segment:
        segments.append('\n'.join(current_segment))

    return segments
