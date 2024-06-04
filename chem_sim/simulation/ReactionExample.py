import numpy as np
from database import chemical_db, reaction_db
from simulator import ChemicalReactionSimulator, OrganicReactionSimulator

print('\n---IONIC REACTION---')
# Initialize the chemical reaction simulator
simulator = ChemicalReactionSimulator(reaction_db, chemical_db)

# Define input reactants and their amounts for the ionic reaction
input_reactants = ['FeO', 'H^+', 'MnO4^-', 'K^+', 'Cl^-']
input_reactant_mol = np.array([2, 6, 0.25, 0.25, 6])
input_solvent_vol = 3

# Create a dictionary of reactants and their molar amounts
reactants_dict = {reactant: mol for reactant, mol in zip(input_reactants, input_reactant_mol)}

# Simulate the chemical reaction
result = simulator.simulate_reaction(reactants_dict, input_solvent_vol)
print('(Components Interface) result:', result)

# Get detailed information about the reaction result
info = simulator.get_information(result)
print('\n(Properties Interface) info:', info)

print('\n---ORGANIC REACTION---')
# Initialize the organic reaction simulator with a specified temperature
sim = OrganicReactionSimulator(tem=200)

# Define input reactants and their amounts for the organic reaction
reactants_dict = {'BrBr': 10, 'c1ccc2cc3ccccc3cc2c1': 20}

# Simulate the organic reaction
result, products = sim.simulate_reaction(reactants_dict)
print('(Components Interface) result:', result, products)

# Get detailed information about the organic reaction result
state, info = sim.get_information()
print('(Properties Interface) state:', state)
print('info:')
for key, value in info[0].items():
    print(f" {key}: {value}")
