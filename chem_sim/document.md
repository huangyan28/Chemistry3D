## 1 Getting Start
## 1.1 `Examples`

### 1.1.1 Inorganic Part

- The IONIC REACTION section in `inorganic_example.py` provides examples
  - Initialization: Set up the simulator `ChemicalReactionSimulator`
  - Input: Construct a reactants dictionary `reactants_dict` {name: moles}
  - Product Prediction: `.simulate_reaction` method, returns a dictionary
  - Characterization Calculation: `.get_information` method, returns a dictionary
### 1.1.2 Organic Part
- The ORGANIC REACTION section in `organic_example.py` provides examples
  - Initialization: Set up the simulator `OrganicReactionSimulator`
  - Input: Construct a reactants dictionary `reactants_dict` {name: moles}
  - Product Prediction: `.simulate_reaction` method, returns: a products dictionary
  - Characterization Calculation: `.get_information` method, returns a products state dictionary {name: state}, a list of product detailed information dictionaries [{property: content},...]

## 1.2 Limitations and Debug

### 1.2.1 Inorganic Part

- Incorrect simulation components, please consider in order:
  - Missing reactants in `database.py - chemicals`
  - Missing reaction equations in `database.py - reactions`
  - Adjust the reaction order in `database.py - reactions`

- Incorrect enthalpy change calculation, please consider in order:
  - Missing or incorrect enthalpy change information for reactants in `database.py - chemicals`
  - Refer to 2.2.1 `simulate_reaction` method

- Incorrect temperature calculation, please consider in order:
  - Incorrect enthalpy change calculation
  - Temperature can only be expressed in the range [0,100]; increase the amount of solvent `input_solvent_vol` to reduce temperature change
  - Refer to 2.2.1 `simulate_reaction` method

- Incorrect pH calculation, please consider in order:
  - Incorrect temperature calculation, leading to incorrect `ionization_constant` calculation
  - Refer to 2.2.1 `ph_calculation` method

- Incorrect color calculation, please consider in order:
  - Limitation: only considers color mixing, not the formation of complexes and chemical equilibria
  - Color information of substances in the database `database.py - chemicals`
  - Color formation of substances at a certain concentration: 2.2.1 `color_dilution` method
  - Color mixing function: `util.py - mix_color`
  - Global transparency adjustment: 2.2.1 `color_dilution` method, `K_const` can change overall transparency

### 1.2.2 Organic Part

- States do not consider gases
- Product mass is calculated as crude yield
- For more information on substances, use cas_number for queries




## 2 Code Explanation
## 2.1 Database `database.py`

### 2.1.1 Data Structure: `Chemical`

The `Chemical` class is used to describe the properties of chemical substances.

#### Attributes

- `name`: Name of the chemical substance
- `color`: Color (optional)
- `state`: State (e.g., gas, liquid, solid, etc.)
- `enthalpy`: Enthalpy value (unit: J/mol)

#### Constructor

```python
def __init__(self, name, color=None, enthalpy=None, state=None):
    # Initialize the attributes of the chemical substance
```
 
### 2.1.2 Data Structure: `Reaction`

The `Reaction` class is used to represent ionic reactions.

#### Attributes

- `reactants`: List of reactants
- `products`: List of products
- `mole_ratio`: Dictionary of mole ratios (example: {'reactant_A': 2, 'reactant_B': 1, 'product_C': 3})
- `enthalpy`: Enthalpy change of the reaction (unit: J)

#### Constructor

```python
def __init__(self, reactants, products, mole_ratio):
    # Initialize the attributes of the ionic reaction
```

### 2.1.3 Database Class: ChemicalDatabase

The `ChemicalDatabase` class is used to manage a collection of chemical substances and provide lookup functionality.

#### Methods

- `find(chemical_name)`: Find the corresponding chemical object by name <br> `return`: `Chemical` object or `None`

### 2.1.4 Database Class: ReactionDatabase

The `ReactionDatabase` class is used to manage a collection of ionic reactions and provide functionality to lookup reactions based on reactants.

#### Methods

- `get_reaction(reactants)`: Find the corresponding ionic reaction object by reactants <br> `return`: `Reaction` object or `None`

## 2.2 Simulator `simulator.py`

### 2.2.1 Simulator Class: ChemicalReactionSimulator

The `ChemicalReactionSimulator` class is used to simulate ionic reaction processes and provides functionality to calculate the color and pH value of the mixed solution.

#### Methods

#### `simulate_reaction(input, input_sol_vol)`
- Simulate Reaction
    - This method is used to simulate the ionic reaction process. It takes two parameters:
    - `input`: A dictionary containing the moles of reactants.
    - `input_sol_vol`: A list containing the volume of solvents.

    **Implementation Logic:**
    - Loop the simulation process: This method repeatedly executes the following steps until equilibrium is reached or specific conditions trigger an exit from the loop.
    - Equilibrium Judgement: Use `charge_balance()` to determine if the charges of the reactants are balanced. If unbalanced, return "Unbalance Charge".
    - Reaction Type Determination: Based on the type of reactants, obtain the corresponding reaction information from the ionic reaction database.
    - Initialize Reactants and Products: Initialize the substances involved in the reaction and their mole numbers according to the mole ratio, and calculate the mole numbers in a 1 mol equation.
    - Update Moles of Reactants and Products: Update the moles of the substances involved in the reaction and the products based on the mole ratio of the reaction.
    - Enthalpy and Temperature Calculation: Calculate the enthalpy change and temperature change based on the enthalpy change and mole numbers of the reaction, and update the temperature using `temperature()`.
    - Simulation End Condition: If the simulation result is the same as the previous result, end the simulation; if the temperature is not within the range of 0~100 degrees, calculate it as 0 or 100.

    **Return Value:**
    - A dictionary of reaction products after simulation or specific information (e.g., "Unbalance Charge").

#### `mix_solutions(products)`

- Calculate Color
    - `products`: A dictionary containing the moles of products.

    **Implementation Logic:**
    - Mix Solutions: Based on the given products, call the `color_dilution()` method for each substance to obtain the color at the current concentration.
    - Color Calculation: Use the `mix_color()` method to calculate the final color based on the mixed solution colors.

    **Return Value:**
    - An rgba list containing the color of the mixed solution.


#### `color_dilution(org_name, org_conc)`

- Calculate rgba color
    - `org_name`: Name of a substance
    - `org_conc`: Concentration of a substance

    **Implementation Logic:**
    - Distinguish between the opacity of three states of matter
    - Use the formula `a = 1 - 10 ** (-K_const * add_conc)` to calculate the opacity of liquids at different concentrations
    - `K_const` can change the overall opacity

    **Return Value:**
    - rgba list

#### `get_information(products)`

- Output all characterization results
    - `products`: A dictionary containing the moles of products.

    **Implementation Logic:**
    - Output `enthalpy`,`color`,`temperature`,`ph`,`volume`,`state` in turn

    **Return Value:**
    - A dictionary containing the characterization of the products


#### `ph_calculation(product)`

- Calculate pH
    - `products`: A dictionary containing the moles of products.

    **Implementation Logic:**
    - Determine based on products: Determine whether the product is acidic or alkaline.
    - Ionization constant calculation: Calculate the ionization constant of water based on the current temperature `temperature` by calling `ionization_constant(tem)`.
    - pH calculation: Calculate the pH value based on the acidity or alkalinity of the substance.
    - Return pH value: Return the calculated pH value.

    **Return Value:**
    - The calculated pH value.

#### `reset()`

- After each simulation, reset the attributes contained in the simulator


### 2.2.2 Simulator Class: OrganicReactionSimulator

The `OrganicReactionSimulator` class is used to simulate organic reaction processes and can predict product generation and query product properties.

#### Methods

#### `__init__(self)`

- Initialization method. Set the RXN4Chemistry API key and initialize the required variables.

#### `simulate_reaction(input)`

- Used to simulate organic reactions.
  - `input`: A dictionary containing SMILES symbols as keys and masses as values.

  **Implementation Logic:**
  - Calculate the total mass of reactants `reactant_mass`.
  - Initiate a request for chemical reaction prediction.
  - Parse the response and extract product information `self.formula`.
  - Use the predicted yield `chanlv` to calculate the moles of products `self.product_mass`.

   **Return Value:**
  - A dictionary of products and their masses.

#### `get_info()`

- Method to obtain product information.
  - Call `get_info_organic()` to obtain information related to the products and return it.

  **Implementation Logic:**
  - Query the SMILES of the product on the website `https://www.chemspider.com/Search.aspx?q={name}`.
  - Crawl the substance information of Properties - Predicted-ACD/Labs and extract it into a dictionary.
  - Supplement with name, molecular formula, SMILES, cas_number, melting point.

  **Return Value:**
  - Dictionary containing product state and characterization.