import json
import numpy as np
from typing import List

def cal_delta_enthalpy(reaction_file, material_file):
    """calculate delta enthalpy of specific reactions

    Args:
        reaction_file (str): reaction information file path
        material_file (str): file path of all available material information

    Returns:
        float: delta enthalpy of the reaction
    """
    
    with open(reaction_file,'r') as f:
        reaction_dict = json.load(f)    
    with open(material_file,'r') as f:
        material_dict = json.load(f)
    
    reactants = reaction_dict['REACTANTS']
    products = reaction_dict['PRODUCTS']
    reactants_enthalpy = np.array([material_dict[x]['norm_enthalpy'] for x in reactants])
    products_enthalpy = np.array([material_dict[x]['norm_enthalpy'] for x in products])
    enthalpy_arr = np.concatenate([reactants_enthalpy,products_enthalpy],-1)
    coeff = np.array(reaction_dict['conc_coeff_arr']).reshape(enthalpy_arr.shape)
    return (enthalpy_arr * coeff).sum()

# def cal_resolution_capacity(mass, capacities):
#     return capacities * mass / mass.sum()

def cal_delta_T(Q:float, materials:List[str], amounts:np.array, capacity_file:str):
    with open(capacity_file,'r') as f:
        capacity_dict = json.load(f)
    mol_capacities = np.array([capacity_dict[m] for m in materials])
    return - Q / (amounts * mol_capacities).sum()

if __name__ == '__main__':
    print(cal_delta_enthalpy('Fe&Mn.json','materials.json'))