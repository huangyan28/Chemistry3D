# Chemistry3D: Robotic Interaction Benchmark for Chemistry Experiments 🧪
## Welcome to Chemistry3D
Chemistry3D is an advanced chemistry simulation laboratory leveraging the capabilities of IsaacSim. This platform offers a comprehensive suite of both organic and inorganic chemical experiments, which can be conducted in a simulated environment. One of the distinguishing features of Chemistry3D is its ability to facilitate robotic operations within these experiments. The primary objective of this simulation environment is to provide a robust and reliable testing ground for both experimental procedures and the application of robotics within chemical laboratory settings.

The development of this system aims to integrate chemical experiment simulation with robotic simulation environments, thereby enabling the visualization of chemical experiments while facilitating the simulation of realistic robotic operations. This environment supports the advancement of embodied intelligence for robots, the execution of reinforcement learning tasks, and other robotic operations. Through this integrated environment, we aspire to seamlessly combine chemical experiment simulation with robotic simulation, ultimately creating a comprehensive test environment for the entire chemical experimentation process.

Contact me at eis_hy@whu.edu.cn

* **The first thing you need to do to develop and run demos for robotics operations is to make sure that Issac-Sim is already installed on your operating device.**
* [**Issac—Sim**](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html)See more details in this link
* [**Chemistry3D Homepage**](https://www.omni-chemistry.com/#/)See more details in this link
* [**Chemistry3D Document**](https://www.omni-chemistry.com/#/)See more details in this link

# How to Download Chemistry3D

Follow the steps below to download the repository, navigate to the correct directory, and run the demo.

## Step 1: Clone the Repository

First, you need to clone the repository from GitHub. Open your terminal and run the following command:

```bash
git clone https://github.com/WHU-DOUBLE/Chemistry3D.git
cd Chemistry3D
pip install -r requirements.txt

## Robotic Manipulation
**To run the demo of chemistry experiment, follow these steps:**



* 
- Example provided in the `inorganic_example.py` under the IONIC REACTION section:
  - Initialization: Set up the simulator `ChemicalReactionSimulator`.
  - Input: Build a reactants dictionary `reactants_dict` {Name: Moles}.
  - Product Prediction: `.simulate_reaction` method, returns a dictionary {Name: Moles}.
  - Characterization Calculation: `.get_information` method, returns a dictionary {Property: Content}.

### 1.2 Organic Part

- Example provided in the `organic_example.py` under the ORGANIC REACTION section:
  - Initialization: Set up the simulator `OrganicReactionSimulator`.
  - Input: Build a reactants dictionary `reactants_dict` {Name: Mass}.
  - Product Prediction: `.simulate_reaction` method, returns a dictionary of products {Name: Mass}.
  - Characterization Calculation: `.get_information` method, returns: a dictionary of product states {Name: State}, a list of dictionaries containing detailed product information [{Property: Content}, ...].

### 1.3 Program Logic

- `Class ChemicalReactionSimulator` Ionic Reaction
  - `simulate_reaction(input, input_sol_vol)`: Simulates the process of ionic reactions
    - `Database.py`: Inorganic information database
      - `ChemicalDatabase`: Manages a collection of chemical substances
      - `ReactionDatabase`: Manages a collection of ionic reactions
    - `charge_balance()`: Determines the charge balance of reactants
    - `temperature()`: Enthalpy and temperature calculation
  - `mix_solutions(products)`: Calculate color
    - `color_dilution()`: Calculate the color of substances at different concentrations
    - `mix_color()`: Mixed solution color
  - `ph_calculation(product)`: Calculate pH
    - `ionization_constant(tem)`: Calculate the ionization constant of water
  - `get_information(products)`: Output all characterization results
    - Contains `enthalpy`, `color`, `temperature`, `ph`, `volume`, `state`
  - `reset()`: Reset reactant information

- `Class OrganicReactionSimulator` Organic Reaction
  - `simulate_reaction(input)`: Simulate organic reactions
    - `rxn4chemistry package`
  - `get_info()`: Get detailed information about the products
    - `get_info_organic()`: Web scraping

