import copy
import math
import sys
import numpy as np
import time

from rxn4chemistry import RXN4ChemistryWrapper
from chem_sim.simulation.utils import *
from chem_sim.simulation.SearchInfo import *
from chem_sim.simulation.database import chemical_db, reaction_db


def __init__(self, solute=None, org=False, volume=0, temp=25, verbose=False):
    if solute is None:
        solute = {}
    self.solute = solute
    self.temp = temp
    self.volume = volume


class Container:
    """
    A container class for simulating chemical reactions. This class handles both organic and inorganic reactions,
    managing solute information, volume, temperature, and interaction with the simulator.

    Args:
        solute (dict, optional): Dictionary of solute substances and their quantities. Defaults to None.
        org (bool, optional): Flag to indicate if the container is organic. Defaults to False.
        volume (float, optional): Volume of the container in liters. Defaults to 0.
        temp (float, optional): Temperature of the container in degrees Celsius. Defaults to 25.
        verbose (bool, optional): Flag for verbose output. Defaults to False.
    """

    def __init__(self, solute=None, org=False, volume=0, temp=25, verbose=False):\
        if solute is None:
            solute = {}
        self.solute = solute
        if volume == 's': 
            volume = 0.000001
        self.volume = volume
        self.temp = temp
        self.org = org
        if self.org:
            self.simulator = None
            self.product = None
        else:
            self.simulator = ChemicalReactionSimulator(reaction_db, chemical_db, verbose=verbose)

    def update(self, new_container, new_volume=None):
        """
        Updates the current container with solute and volume from another container. Handles both organic and inorganic cases.

        Args:
            new_container (Container): The container to merge into the current container.
            new_volume (float, optional): The volume to transfer from the new container. Defaults to None.

        Returns:
            list: A list of solute concentrations over time.
        """
        new_org = new_container.org
        if new_org != self.org:
            sys.exit('Mismatched Container Type')

        if self.org:
            # For organic container
            new_solute = new_container.solute
            if new_volume is None:
                new_volume = copy.deepcopy(new_container.volume)
            self.volume += new_volume
            for key, value in new_solute.items():
                if key in self.solute:
                    self.solute[key] += value / new_container.volume * new_volume
                else:
                    self.solute[key] = value / new_container.volume * new_volume

            old_solute = {}
            for key, value in new_container.solute.items():
                updated_value = value - value / new_container.volume * new_volume
                if updated_value != 0:
                    old_solute[key] = updated_value
            new_container.solute = old_solute
            new_container.volume -= new_volume

            num = len(list(self.solute.keys()))
            mix_solute = copy.deepcopy(self.solute)
            concentrations_list = [mix_solute]
            if num > 1:
                self.simulator = OrganicReactionSimulator()
                self.solute, self.product = self.simulator.simulate_reaction(self.solute)
                concentrations_list = simulate_time_org(mix_solute, self.solute, vol=self.volume)

        else:
            # For inorganic container
            new_solute = new_container.solute
            if new_volume is None:
                new_volume = copy.deepcopy(new_container.volume)
            self.temp = (self.temp * self.volume + new_volume * new_container.temp) / (new_volume + self.volume)
            self.volume += new_volume
            for key, value in new_solute.items():
                if key in self.solute:
                    self.solute[key] += value / new_container.volume * new_volume
                else:
                    self.solute[key] = value / new_container.volume * new_volume

            old_solute = {}
            for key, value in new_container.solute.items():
                updated_value = value - value / new_container.volume * new_volume
                if updated_value != 0:
                    old_solute[key] = updated_value
            new_container.solute = old_solute
            new_container.volume -= new_volume
            new_container.solute = new_container.simulator.simulate_reaction(new_container.solute, new_container.volume)

            mixed = self.solute.copy()
            reaction = reaction_db.get_reaction(mixed)
            if reaction is None:
                concentrations_list = [mixed]
            else:
                filtered_dict = {key: value for key, value in mixed.items() if value != 0}
                concentrations_list = simulate_time(filtered_dict, reaction, vol=self.volume)

            self.solute = self.simulator.simulate_reaction(self.solute, self.volume, temp=self.temp)
            self.temp = self.simulator.get_temperature()

        return concentrations_list

    def get_info(self, verbose=False):
        """
        Retrieves detailed information about the current state of the container.

        Args:
            verbose (bool, optional): If True, prints detailed information. Defaults to False.

        Returns:
            list: A list containing solute information and additional properties.
        """
        if self.org:
            info = [self.solute, get_info_organic(self.product[0])]
        else:
            info = [self.solute, self.simulator.get_information(self.solute)]
        if verbose:
            print('(Solute info):', info[0])
            print('(Property info):', info[1])
        return info


def simulate_time(initial_state, reaction, vol, dt=1/60, rate=None):
    """
    Simulates the reaction over time, returning the concentration of reactants and products at each time step.

    Args:
        initial_state (dict): Initial concentrations of reactants.
        reaction (Reaction): The reaction being simulated.
        vol (float): Volume of the solution.
        dt (float, optional): Time step for simulation. Defaults to 1/60.
        rate (float, optional): Reaction rate. Defaults to None.

    Returns:
        list: A list of concentration dictionaries for each time step.
    """
    concentrations_list = []
    concentrations = initial_state.copy()
    for product in reaction.products:
        concentrations[product] = 0
    concentrations_list.append(concentrations.copy())

    if rate is None:
        try:
            rate = reaction.rate
        except:
            rate = 0.15
    reaction_aq = [reactant for reactant in reaction.reactants if chemical_db.find(reactant).state == 'aq']

    flag = 0
    while True:
        velocity = rate
        sq = 0
        for reactant in reaction_aq:
            velocity *= (concentrations[reactant] / vol) ** reaction.mole_ratio[reactant]
            sq += reaction.mole_ratio[reactant]
        velocity = math.pow(velocity, 1/sq)

        limiting_reactant = min(reaction.reactants, key=lambda x: concentrations[x] / reaction.mole_ratio[x])
        max_velocity = concentrations[limiting_reactant] / (reaction.mole_ratio[limiting_reactant] * dt)
        velocity = max(velocity, 0.01 / (reaction.mole_ratio[limiting_reactant] * dt))
        if velocity > max_velocity:
            velocity = max_velocity
            flag = 1

        for reactant in reaction.reactants:
            concentrations[reactant] += -velocity * reaction.mole_ratio[reactant] * dt

        for product in reaction.products:
            concentrations[product] += velocity * reaction.mole_ratio[product] * dt

        concentrations_output = {key: round(value, 5) for key, value in concentrations.items()}
        concentrations_list.append(concentrations_output)
        if flag:
            break

    return concentrations_list

def simulate_time_org(initial_state, output, vol, dt=1/60, rate=0.015):
    """
    Simulates the reaction for organic compounds over time, returning the concentration of reactants and products at each time step.

    Args:
        initial_state (dict): Initial concentrations of reactants.
        output (dict): Expected final concentrations of products.
        vol (float): Volume of the solution.
        dt (float, optional): Time step for simulation. Defaults to 1/60.
        rate (float, optional): Reaction rate. Defaults to 0.015.

    Returns:
        list: A list of concentration dictionaries for each time step.
    """
    concentrations_list = []
    concentrations = copy.deepcopy(initial_state)
    for product in output.keys():
        if product not in initial_state:
            concentrations[product] = 0
    concentrations_list.append(concentrations.copy())

    flag = 0
    sq = len(initial_state.keys())

    while True:
        velocity = rate
        for reactant in initial_state.keys():
            velocity *= (concentrations[reactant] / vol)
        velocity = math.pow(velocity, 1/sq)

        for reactant in concentrations.keys():
            if reactant in initial_state.keys():
                concentrations[reactant] -= velocity * dt
            else:
                concentrations[reactant] += velocity * dt * sq
                if concentrations[reactant] >= output[reactant]:
                    concentrations_list.append(copy.deepcopy(output))
                    flag = 1
                    break

        if flag:
            break

        concentrations_output = {key: round(value, 5) for key, value in concentrations.items()}
        concentrations_list.append(concentrations_output)

    return concentrations_list


class ChemicalReactionSimulator:
    """
    A simulator for inorganic chemical reactions. Handles the simulation of reactions, updating temperatures,
    volumes, and generating concentration data over time.

    Args:
        re_db (ReactionDatabase): Database of possible reactions.
        ch_db (ChemicalDatabase): Database of chemical properties.
        verbose (bool, optional): Flag for verbose output. Defaults to True.
    """

    def __init__(self, re_db, ch_db, verbose=True):
        self.re_db = re_db
        self.ch_db = ch_db
        self.enthalpy = 0
        self.volume = 0
        self.tem = 25
        self.state = set()
        self.flag = 1
        self.verbose = verbose

    def simulate_reaction(self, input, input_sol_vol, temp=25):
        """
        Simulates a chemical reaction given the input substances and volume.

        Args:
            input (dict): Dictionary of reactants and their quantities.
            input_sol_vol (float): Volume of the solution.
            temp (float, optional): Temperature of the reaction. Defaults to 25.

        Returns:
            dict: Resulting concentrations of substances after the reaction.
        """
        self.tem = temp
        previous_result = input.copy()
        if self.verbose:
            print('current result:', previous_result)
        while True:
            input_re = list(input.keys())
            input_re_amounts = list(input.values())

            if round(charge_balance(input_re, input_re_amounts), 4) != 0:
                return "Unbalance Charge"

            for rea in input_re:
                rea0 = self.ch_db.find(rea)
                if rea0 is None:
                    print("no chemical in database:", rea)
                    self.flag = 0

            self.volume = round(input_sol_vol, 5)
            reaction = self.re_db.get_reaction(input_re)
            if reaction is None:
                if self.verbose:
                    print("No reaction in database\n")
                return input
            if self.verbose:
                print("reacting:", reaction.reactants)

            reacting = {reactant: amount for reactant, amount in zip(input_re, input_re_amounts)}

            unreacting = {key: reacting.pop(key) for key in list(reacting.keys()) if key not in reaction.reactants}

            producing = {product: 0 for product in reaction.products}

            reacting1mol = {key: value / reaction.mole_ratio[key] for key, value in reacting.items()}
            mol = min(reacting1mol.values())

            self.enthalpy = -reaction.enthalpy * mol
            self.tem = temperature(self.volume, reaction.enthalpy * mol, self.tem)

            for idx, reactant in enumerate(reaction.reactants):
                reacting[reactant] -= mol * reaction.mole_ratio[reactant]

            for idx, product in enumerate(reaction.products):
                producing[product] = mol * reaction.mole_ratio[product]

            r1 = copy.deepcopy(reacting)
            r1.update(producing)
            for key, value in unreacting.items():
                if key in r1:
                    r1[key] += value
                else:
                    r1[key] = value
            r1.pop('H2O', None)
            rounded_r1 = {key: round(value, 5) for key, value in r1.items() if value != 0}

            if rounded_r1 == previous_result:
                if self.tem < 0:
                    self.tem = 0
                elif self.tem > 100:
                    self.tem = 100
                break

            input = rounded_r1
            if self.verbose:
                print('current result:', input)

        return input

    def get_enthalpy(self):
        """
        Returns the enthalpy change of the reaction.

        Returns:
            float: Enthalpy change in kJ.
        """
        return self.enthalpy

    def get_temperature(self):
        """
        Returns the temperature of the solution after the reaction.

        Returns:
            float: Temperature in degrees Celsius.
        """
        return round(self.tem, 2)

    def get_volume(self):
        """
        Returns the volume of the solution.

        Returns:
            float: Volume in liters.
        """
        return self.volume

    def color_dilution(self, org_name, org_conc):
        """
        Calculates the color dilution based on the concentration of the organic substance.

        Args:
            org_name (str): Name of the organic substance.
            org_conc (float): Concentration of the organic substance.

        Returns:
            list: RGBA color values.
        """
        if org_conc == 0:
            return None
        rgb = self.ch_db.find(org_name).color
        if rgb is None:
            return None
        rgb = [i for i in rgb]
        a = 0.1  # Transparency for gases
        state = self.ch_db.find(org_name).state
        if state == 'aq':
            K_const = 3
            add_conc = org_conc / self.volume
            a = 1 - 10 ** (-K_const * add_conc)
        elif state == 's':
            a = 1
        rgba = rgb + [a]
        return rgba

    def mix_solutions(self, products):
        """
        Mixes solutions and returns the resulting color.

        Args:
            products (dict): Dictionary of products and their quantities.

        Returns:
            list: RGBA color values of the mixed solution.
        """
        color = [0, 0, 0, 0]
        for key, value in products.items():
            color_new = self.color_dilution(key, value)
            if color_new is not None:
                color = mix_color(color_new, color)
        co = [round(num) if idx < 3 else round(num, 4) for idx, num in enumerate(color)]
        return co

    def ph_calculation(self, product):
        """
        Calculates the pH of the solution based on the product concentrations.

        Args:
            product (dict): Dictionary of products and their concentrations.

        Returns:
            float: pH value of the solution.
        """
        kw = ionization_constant(self.tem)
        if 'H^+' in product:
            H_concentration = product.get('H^+', 0) / self.volume
            pH = -np.log10(H_concentration)
            return pH
        elif 'OH^-' in product:
            OH_concentration = product.get('OH^-', 0) / self.volume
            pOH = -np.log10(OH_concentration)
            pH = 14 - pOH
            return pH
        else:
            return round(-np.log10(kw) / 2 + 7, 2)

    def get_information(self, result):
        """
        Retrieves detailed information about the resulting solution.

        Args:
            result (dict): Dictionary of products and their concentrations.

        Returns:
            dict: A dictionary containing properties of the resulting solution.
        """
        if isinstance(result, dict):
            output = {}
            enthalpy_change = self.enthalpy
            output['enthalpy'] = enthalpy_change
            rgba_color = self.mix_solutions(result)
            output['color'] = rgba_color
            tem = self.get_temperature()
            output['temperature'] = tem
            ph = self.ph_calculation(result)
            output['ph'] = ph
            vol = self.volume
            output['volume'] = vol
            for key in result.keys():
                self.state.add(self.ch_db.find(key).state)
            output['state'] = list(self.state)
            if list(self.state) == ['s']:
                output['volume'] = 0
            if self.verbose:
                print("\nProperties of result:")
                print(f'Enthalpy Change:\t{enthalpy_change} kJ')
                print(f'RGBA Color:\t\t\t{rgba_color}')
                print(f'Temperature:\t\t{tem} °C')
                print(f'pH:\t\t\t\t\t{ph}')
                print(f'Volume:\t\t\t\t{vol} L')
                print(f'State:\t\t\t\t{self.state}')
            self.state = set()
            return output

    def reset(self):
        """
        Resets the simulator to its initial state.
        """
        self.enthalpy = 0
        self.volume = 0
        self.tem = 25
        self.state = set()
        self.flag = 1


class OrganicReactionSimulator:
    """
    A simulator for organic chemical reactions using the RXN4Chemistry API.

    Args:
        tem (float, optional): Temperature for the reaction. Defaults to None.
    """

    def __init__(self, tem=None):
        api_key = 'apk-22af09d4f8e936a1ec0800bc7abef107622a00c4933a257cb43ad56aaf667f4d5b892bdcb28be2c848e1fcb1cea4b459d46c059138f933224beef6d7f978bc70e8da3d2b1fb2aa02cee8945a461e21ae'
        print('Initializing...')
        self.rxn4chemistry_wrapper = RXN4ChemistryWrapper(api_key=api_key)
        self.rxn4chemistry_wrapper.create_project('organic')
        time.sleep(2)
        self.tem = tem
        self.formula = None
        self.reactant = None
        self.product = None
        self.reactant_mass = 0
        self.product_mass = 0

    def simulate_reaction(self, inputt):
        """
        Simulates an organic chemical reaction given the input substances.

        Args:
            inputt (dict): Dictionary of reactants and their quantities.

        Returns:
            dict: Resulting concentrations of substances after the reaction.
            list: List of products generated.
        """
        smiles_list = list(inputt.keys())
        self.reactant_mass = sum(list(inputt.values()))
        self.reactant = smiles_list
        reactants = '.'.join(smiles_list)
        print('Reacting...')
        response = self.rxn4chemistry_wrapper.predict_reaction(reactants)
        results = self.rxn4chemistry_wrapper.get_predict_reaction_results(response['prediction_id'])
        self.formula = results['response']['payload']['attempts'][0]['smiles']
        self.product = self.formula.split('>>')[1].split()
        time.sleep(2)
        print('Reaction:', self.formula)

        response2 = self.rxn4chemistry_wrapper.predict_reaction_properties(
            reactions=[self.formula],
            ai_model="yield-2020-08-10",
        )
        chanlv = float(response2["response"]["payload"]["content"][0]["value"])

        self.product_mass = self.reactant_mass * chanlv / 100
        time.sleep(2)
        output = {pro: round(self.product_mass, 4) for pro in self.product}

        for key in inputt.keys():
            output[key] = round((1-chanlv/100) * inputt[key], 4)
        return output, self.product

    def get_information(self):
        """
        Retrieves detailed information about the resulting organic reaction.

        Returns:
            dict: State information and detailed properties of the products.
        """
        info = []
        state = {}
        for p in self.product:
            new_info = get_info_organic(p)
            info.append(new_info)
            melt = new_info['melting_point']
            new_state = 'l'
            if melt > self.tem:
                new_state = 's'
            state[p] = new_state

        return state, info

# class Container_org:
#     def __init__(self, solute=None, org=False, volume=0, temp=25, verbose=False):
#         if solute is None:
#             solute = {}
#         self.solute = solute
#         self.temp = temp
#         self.volume = volume
#         self.org = org
#         if self.org:
#             self.simulator = None
#             self.product = None
#
#
#     def update(self, new_container, new_volume=None):
#         # 当前烧杯
#         new_solute = new_container.solute
#         if new_volume is None:
#             new_volume = copy.deepcopy(new_container.volume)
#         self.volume += new_volume
#         for key, value in new_solute.items():
#             if key in self.solute:
#                 self.solute[key] += value / new_container.volume * new_volume
#             else:
#                 self.solute[key] = value / new_container.volume * new_volume
#
#         # 倒进来的烧杯
#         old_solute = {}
#         for key, value in new_container.solute.items():
#             updated_value = value - value / new_container.volume * new_volume
#             if updated_value != 0:
#                 old_solute[key] = updated_value
#         new_container.solute = old_solute
#         new_container.volume -= new_volume
#
#         # 反应
#         num = len(list(self.solute.keys()))
#         if num > 1:
#             self.simulator = OrganicReactionSimulator()
#             self.solute, self.product = self.simulator.simulate_reaction(self.solute)
#
#     def get_info(self, verbose=False):
#         info = [self.solute,get_info_organic(self.product[0])]
#         if verbose:
#             print('(组分接口 info[0])：', info[0])
#             print('(表征接口 info[1])：', info[1])
#         return info


