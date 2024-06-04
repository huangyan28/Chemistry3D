<<<<<<< HEAD
# Chemistry3D 🧪
## 1 Getting Start

### 1.1 Inorganic Part

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
=======
# Chemistry3D
>>>>>>> 620cea4ffa2e14fc531029a7ff960bcce45f8f58
