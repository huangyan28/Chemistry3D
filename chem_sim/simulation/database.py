import numpy as np

# Define the Chemical class to represent a chemical substance
class Chemical:
    def __init__(self, name, color=None, enthalpy=None, state=None):
        """
        Initialize a Chemical object.

        Args:
            name (str): The name of the chemical.
            color (tuple, optional): The color of the chemical in RGB format. Defaults to None.
            enthalpy (float, optional): The enthalpy of the chemical in kJ/mol. Defaults to None.
            state (str, optional): The state of the chemical ('s', 'l', 'g', 'aq'). Defaults to None.
        """
        self.name = name
        self.color = color
        self.enthalpy = enthalpy
        self.state = state

# Define the ChemicalDatabase class to store and retrieve chemical information
class ChemicalDatabase:
    def __init__(self, chemicals):
        """
        Initialize a ChemicalDatabase object.

        Args:
            chemicals (dict): A dictionary of Chemical objects indexed by their names.
        """
        self.chemicals = chemicals

    def find(self, chemical_name):
        """
        Find and return a Chemical object by its name.

        Args:
            chemical_name (str): The name of the chemical.

        Returns:
            Chemical: The Chemical object if found, otherwise None.
        """
        chemical = self.chemicals.get(chemical_name)
        if chemical:
            return chemical
        return None

# Initialize the chemical database with various chemicals
chemicals = {
    'H^+': Chemical('H^+', color=None, enthalpy=0, state='aq'),
    'OH^-': Chemical('OH^-', color=None, enthalpy=-229.9, state='aq'),
    'H2O': Chemical('H2O', color=None, enthalpy=-285.830, state='l'),
    'Co^2+': Chemical('Co^2+', color=(207, 104, 140), enthalpy=-58.2, state='aq'),
    'Co(OH)2': Chemical('Co(OH)2', color=(173, 216, 230), enthalpy=-539.7, state='s'),
    'Fe^3+': Chemical('Fe^3+', color=(255, 255, 153), enthalpy=-47.7, state='aq'),
    'SCN^-': Chemical('SCN^-', color=None, enthalpy=76.44, state='aq'),
    'Fe(SCN)3': Chemical('Fe(SCN)3', color=(150, 0, 24), enthalpy=None, state='s'),
    'Fe^2+': Chemical('Fe^2+', color=(152, 178, 150), enthalpy=-87.9, state='aq'),
    'SO4^2-': Chemical('SO4^2-', color=None, enthalpy=-907.5, state='aq'),
    'K^+': Chemical('K^+', color=None, enthalpy=-251.2, state='aq'),
    'MnO4^-': Chemical('MnO4^-', color=(128, 0, 128), enthalpy=-518.4, state='aq'),
    'Mn^2+': Chemical('Mn^2+', color=(166, 142, 190), enthalpy=-218.8, state='aq'),
    'Cu^2+': Chemical('Cu^2+', color=(0, 191, 255), enthalpy=64.4, state='s'),
    'Cu(OH)2': Chemical('Cu(OH)2', color=(0, 0, 128), enthalpy=-450.37, state='s'),
    '[CoCl4]^2-': Chemical('[CoCl4]^2-', color=(255, 192, 203), enthalpy=None, state='s'),
    'Fe': Chemical('Fe', color=(192, 192, 192), enthalpy=0, state='s'),
    'Al': Chemical('Al', color=(192, 192, 192), enthalpy=0, state='s'),
    'Al^3+': Chemical('Al^3+', color=None, enthalpy=-524.7, state='aq'),
    'AgCl': Chemical('AgCl', color=(255, 255, 255), enthalpy=-127, state='s'),
    'Ba^2+': Chemical('Ba^2+', color=None, enthalpy=-538.4, state='aq'),
    'BaSO4': Chemical('BaSO4', color=(255, 255, 255), enthalpy=-1465.2, state='s'),
    'Zn^2+': Chemical('Zn^2+', color=None, enthalpy=-152.4, state='aq'),
    'Zn(OH)2': Chemical('Zn(OH)2', color=(255, 255, 240), enthalpy=-642, state='s'),
    'CH3COOH': Chemical('CH3COOH', color=None, enthalpy=-484.93, state='aq'),
    'CH3COO^-': Chemical('CH3COO^-', color=None, enthalpy=-496.4, state='aq'),
    'Mn(OH)2': Chemical('Mn(OH)2', color=(255, 255, 255), enthalpy=-700, state='s'),
    'Fe(OH)2': Chemical('Fe(OH)2', color=(0, 128, 0), enthalpy=-561.7, state='s'),
    'Fe(OH)3': Chemical('Fe(OH)3', color=(139, 69, 19), enthalpy=-824, state='s'),
    'Ca^2+': Chemical('Ca^2+', color=None, enthalpy=-543.0, state='aq'),
    'Ca(OH)2': Chemical('Ca(OH)2', color=(255, 255, 255), enthalpy=-986.09, state='s'),
    'Mg': Chemical('Mg', color=(192, 192, 192), enthalpy=0, state='s'),
    'Mg^2+': Chemical('Mg^2+', color=None, enthalpy=-462.0, state='aq'),
    'Cu': Chemical('Cu', color=(139, 69, 19), enthalpy=0, state='s'),
    'Zn': Chemical('Zn', color=(192, 192, 192), enthalpy=0, state='s'),
    'Ag^+': Chemical('Ag^+', color=None, enthalpy=105.9, state='s'),
    'Ag': Chemical('Ag', color=(255, 255, 255), enthalpy=0, state='s'),
    'H2': Chemical('H2', color=None, enthalpy=0, state='g'),
    'Pb^2+': Chemical('Pb2^+', color=None, enthalpy=1.6, state='aq'),
    'Br^-': Chemical('Br^-', color=None, enthalpy=-120.9, state='aq'),
    '[PbBr4]^2-': Chemical('[PbBr4]^2-', color=None, enthalpy=None, state='aq'),
    'I^-': Chemical('I^-', color=None, enthalpy=-55.9, state='aq'),
    'PbI^+': Chemical('PbI^+', color=None, enthalpy=None, state='aq'),
    'Na2O': Chemical('Na2O', color=(255, 255, 255), enthalpy=-416, state='s'),
    'Na^+': Chemical('Na^+', color=None, enthalpy=-239.7, state='aq'),
    'MgO': Chemical('MgO', color=(255, 255, 255), enthalpy=-601.8, state='s'),
    'CaO': Chemical('CaO', color=(255, 255, 255), enthalpy=-635.5, state='s'),
    'Al2O3': Chemical('Al2O3', color=(255, 255, 255), enthalpy=-1675.7, state='s'),
    'K2O': Chemical('K2O', color=(255, 255, 255), enthalpy=-363.17, state='s'),
    'K+': Chemical('K+', color=None, enthalpy=-251.2, state='aq'),
    'FeO': Chemical('FeO', color=(0, 0, 0), enthalpy=-272.04, state='s'),
    'CuO': Chemical('CuO', color=(0, 0, 0), enthalpy=-155.2, state='s'),
    'ZnO': Chemical('ZnO', color=(255, 255, 255), enthalpy=-348.0, state='s'),
    'PbO': Chemical('PbO', color=(255, 255, 0), enthalpy=-217.9, state='s'),
    'H2O2': Chemical('H2O2', color=None, enthalpy=-191.17, state='aq'),
    'ClO^-': Chemical('ClO^-', color=None, enthalpy=-107.6, state='aq'),
    'Cl^-': Chemical('Cl^-', color=None, enthalpy=-167.4, state='aq'),
    'ClO2^-': Chemical('ClO2^-', color=None, enthalpy=-67.0, state='aq'),
    'HSO4^-': Chemical('HSO4^-', color=None, enthalpy=-907.5, state='aq'),
    'SO3^2-': Chemical('SO3^2-', color=None, enthalpy=-626.22, state='aq'),
    'HSO3^-': Chemical('HSO3^-', color=None, enthalpy=-608.81, state='aq'),
    'O2': Chemical('O2', color=None, enthalpy=0, state='g'),
    'Al(OH)4^-': Chemical('Al(OH)4^-', color=None, enthalpy=-1502.9, state='aq'),
    'SnO2': Chemical('SnO2', color=(255, 255, 255), enthalpy=-580.7, state='s'),
    'Sn^4+': Chemical('Sn^4+', color=None, enthalpy=158.3, state='aq'),
    'Al(OH)3': Chemical('Al(OH)3', color=(255, 255, 255), enthalpy=-1276, state='s'),
    'Ag2O': Chemical('Ag2O', color=(139, 69, 19), enthalpy=-30.6, state='s'),
    'Fe2O3': Chemical('Fe2O3', color=(139, 69, 19), enthalpy=-822.2, state='s'),
}

chemical_db = ChemicalDatabase(chemicals)

no_enthalpy = []

# Define the Reaction class to represent a chemical reaction
class Reaction:
    def __init__(self, reactants, products, mole_ratio):
        """
        Initialize a Reaction object.

        Args:
            reactants (list): List of reactant names.
            products (list): List of product names.
            mole_ratio (dict): Dictionary of reactants and products with their molar ratios.
        """
        self.reactants = reactants
        self.products = products
        self.mole_ratio = mole_ratio
        self.enthalpy = 0
        for reactant in reactants:
            try:
                self.enthalpy += chemical_db.find(reactant).enthalpy * self.mole_ratio[reactant]
            except:
                no_enthalpy.append(reactant)
                if chemical_db.find(reactant) is None:
                    print(reactant)
        for product in products:
            try:
                self.enthalpy -= chemical_db.find(product).enthalpy * self.mole_ratio[product]
            except:
                if chemical_db.find(product) is None:
                    print(product)
                no_enthalpy.append(product)

# Define the ReactionDatabase class to store and retrieve reactions
class ReactionDatabase:
    def __init__(self, reactions):
        """
        Initialize a ReactionDatabase object.

        Args:
            reactions (list): A list of Reaction objects.
        """
        self.reactions = reactions

    def get_reaction(self, reactants):
        """
        Retrieve a reaction that involves the given reactants.

        Args:
            reactants (list): List of reactant names.

        Returns:
            Reaction: The matching Reaction object if found, otherwise None.
        """
        for reaction in self.reactions:
            if set(reactants).issuperset(set(reaction.reactants)):
                return reaction
        return None

# Initialize the reaction database with various reactions
reactions = [
    Reaction(['H^+', 'OH^-'], ['H2O'], {'H^+': 1, 'OH^-': 1, 'H2O': 1}),
    Reaction(['OH^-', 'HSO4^-'], ['SO4^2-', 'H2O'], {'OH^-': 1, 'HSO4^-': 1, 'SO4^2-': 1, 'H2O': 1}),
    Reaction(['CH3COOH', 'OH^-'], ['CH3COO^-', 'H2O'], {'CH3COOH': 1, 'OH^-': 1, 'CH3COO^-': 1, 'H2O': 1}),

    Reaction(['H^+', 'Na2O'], ['Na^+', 'H2O'], {'H^+': 2, 'Na2O': 1, 'Na^+': 2, 'H2O': 1}),
    Reaction(['H^+', 'MgO'], ['Mg^2+', 'H2O'], {'H^+': 2, 'MgO': 1, 'Mg^2+': 1, 'H2O': 1}),
    Reaction(['H^+', 'CaO'], ['Ca^2+', 'H2O'], {'H^+': 2, 'CaO': 1, 'Ca^2+': 1, 'H2O': 1}),
    Reaction(['H^+', 'Al2O3'], ['Al^3+', 'H2O'], {'H^+': 6, 'Al2O3': 2, 'Al^3+': 4, 'H2O': 3}),
    Reaction(['H^+', 'K2O'], ['K^+', 'H2O'], {'H^+': 2, 'K2O': 1, 'K^+': 2, 'H2O': 1}),
    Reaction(['H^+', 'FeO'], ['Fe^2+', 'H2O'], {'H^+': 2, 'FeO': 1, 'Fe^2+': 1, 'H2O': 1}),
    Reaction(['H^+', 'CuO'], ['Cu^2+', 'H2O'], {'H^+': 2, 'CuO': 1, 'Cu^2+': 1, 'H2O': 1}),
    Reaction(['H^+', 'ZnO'], ['Zn^2+', 'H2O'], {'H^+': 2, 'ZnO': 1, 'Zn^2+': 1, 'H2O': 1}),
    Reaction(['H^+', 'PbO'], ['Pb^2+', 'H2O'], {'H^+': 2, 'PbO': 1, 'Pb^2+': 1, 'H2O': 1}),
    Reaction(['H^+', 'SnO2'], ['Sn^4+', 'H2O'], {'H^+': 4, 'SnO2': 1, 'Sn^4+': 2, 'H2O': 2}),
    Reaction(['H^+', 'Ag2O'], ['Ag^+', 'H2O'], {'H^+': 2, 'Ag2O': 1, 'Ag^+': 2, 'H2O': 1}),
    Reaction(['H^+', 'Fe2O3'], ['Fe^3+', 'H2O'], {'H^+': 6, 'Fe2O3': 1, 'Fe^3+': 2, 'H2O': 3}),

    Reaction(['Fe^2+', 'MnO4^-', 'H^+'], ['Fe^3+', 'Mn^2+', 'H2O'], {'Fe^2+': 5, 'MnO4^-': 1, 'H^+': 8, 'Fe^3+': 5, 'Mn^2+': 1, 'H2O': 4}),
    Reaction(['MnO4^-', 'H2O2', 'H^+'], ['Mn^2+', 'O2', 'H2O'], {'MnO4^-': 2, 'H2O2': 5, 'H^+': 6, 'Mn^2+': 2, 'O2': 5, 'H2O': 3}),
    Reaction(['MnO4^-', 'SO3^2-', 'H^+'], ['Mn^2+', 'SO4^2-'], {'MnO4^-': 2, 'SO3^2-': 5, 'Mn^2+': 2, 'SO4^2-': 5, 'H^+': 6}),
    Reaction(['MnO4^-', 'HSO3^-', 'H^+'], ['Mn^2+', 'SO4^2-'], {'MnO4^-': 2, 'HSO3^2-': 5, 'Mn^2+': 2, 'SO4^2-': 5, 'H^+': 1}),
    Reaction(['ClO^-', 'H2O2'], ['Cl^-', 'O2', 'H2O'], {'ClO^-': 2, 'H2O2': 2, 'Cl^-': 2, 'O2': 1, 'H2O': 2}),
    Reaction(['ClO^-', 'Fe^2+', 'H^+'], ['Cl^-', 'Fe^3+', 'H2O'], {'ClO^-': 1, 'Fe^2+': 2, 'H^+': 2, 'Cl^-': 1, 'Fe^3+': 2, 'H2O': 1}),
    Reaction(['ClO^-', 'SO3^2-'], ['Cl^-', 'SO4^2-'], {'ClO^-': 1, 'SO3^2-': 1, 'Cl^-': 1, 'SO4^2-': 1}),
    Reaction(['ClO^-', 'HSO3^-'], ['Cl^-', 'SO4^2-', 'H^+'], {'ClO^-': 1, 'SO3^2-': 1, 'Cl^-': 1, 'SO4^2-': 1, 'H^+': 1}),
    Reaction(['H2O2', 'Fe^2+', 'H^+'], ['Fe^3+', 'H2O'], {'H2O2': 2, 'Fe^2+': 2, 'H^+': 2, 'Fe^3+': 2, 'H2O': 3}),
    Reaction(['SO3^2-', 'H2O2'], ['SO4^2-', 'H2O'], {'SO3^2-': 1, 'H2O2': 1, 'SO4^2-': 1, 'H2O': 1}),
    Reaction(['HSO3^-', 'H2O2'], ['SO4^2-', 'H^+', 'H2O'], {'HSO3^-': 1, 'H2O2': 1, 'SO4^2-': 1, 'H2O': 1, 'H^+': 1}),

    Reaction(['Mg', 'Ag^+'], ['Mg^2+', 'Ag'], {'Mg': 1, 'Ag^+': 2, 'Mg^2+': 1, 'Ag': 2}),
    Reaction(['Mg', 'Cu^2+'], ['Mg^2+', 'Cu'], {'Mg': 1, 'Cu^2+': 1, 'Mg^2+': 1, 'Cu': 1}),
    Reaction(['Mg', 'H^+'], ['Mg^2+', 'H2'], {'Mg': 1, 'H^+': 2, 'Mg^2+': 1, 'H2': 1}),
    Reaction(['Mg', 'Fe^2+'], ['Mg^2+', 'Fe'], {'Mg': 1, 'Fe^2+': 1, 'Mg^2+': 1, 'Fe': 1}),
    Reaction(['Mg', 'Zn^2+'], ['Mg^2+', 'Zn'], {'Mg': 1, 'Zn^2+': 1, 'Mg^2+': 1, 'Zn': 1}),
    Reaction(['Mg', 'Al^3+'], ['Mg^2+', 'Al'], {'Mg': 2, 'Al^3+': 3, 'Mg^2+': 2, 'Al': 3}),
    Reaction(['Al', 'Ag^+'], ['Al^3+', 'Ag'], {'Al': 2, 'Ag^+': 3, 'Al^3+': 2, 'Ag': 3}),
    Reaction(['Al', 'Cu^2+'], ['Cu', 'Al^3+'], {'Al': 2, 'Cu^2+': 3, 'Cu': 3, 'Al^3+': 2}),
    Reaction(['Al', 'H^+'], ['Al^3+', 'H2'], {'Al': 2, 'H^+': 6, 'Al^3+': 2, 'H2': 3}),
    Reaction(['Al', 'Zn^2+'], ['Al^3+', 'Zn'], {'Al': 2, 'Zn^2+': 3, 'Al^3+': 2, 'Zn': 3}),
    Reaction(['Al', 'Fe^2+'], ['Al^3+', 'Fe'], {'Al': 2, 'Fe^2+': 3, 'Al^3+': 2, 'Fe': 3}),
    Reaction(['Zn', 'Ag^+'], ['Zn^2+', 'Ag'], {'Zn': 1, 'Ag^+': 1, 'Zn^2+': 1, 'Ag': 1}),
    Reaction(['Zn', 'Cu^2+'], ['Zn^2+', 'Cu'], {'Zn': 1, 'Cu^2+': 1, 'Zn^2+': 1, 'Cu': 1}),
    Reaction(['Zn', 'H^+'], ['Zn^2+', 'H2'], {'Zn': 1, 'H^+': 2, 'Zn^2+': 1, 'H2': 1}),
    Reaction(['Zn', 'Fe^2+'], ['Zn^2+', 'Fe'], {'Zn': 1, 'Fe^2+': 1, 'Zn^2+': 1, 'Fe': 1}),
    Reaction(['Fe', 'Ag^+'], ['Fe^2+', 'Ag'], {'Fe': 2, 'Ag^+': 2, 'Fe^2+': 2, 'Ag': 2}),
    Reaction(['Fe', 'Fe^3+'], ['Fe^2+'], {'Fe': 1, 'Fe^3+': 2, 'Fe^2+': 3}),
    Reaction(['Fe', 'H^+'], ['Fe^2+', 'H2'], {'Fe': 1, 'H^+': 2, 'Fe^2+': 1, 'H2': 1}),
    Reaction(['Fe', 'Cu^2+'], ['Cu', 'Fe^2+'], {'Fe': 1, 'Cu^2+': 1, 'Cu': 1, 'Fe^2+': 1}),
    Reaction(['Cu', 'Ag^+'], ['Cu^2+', 'Ag'], {'Cu': 1, 'Ag^+': 2, 'Cu^2+': 1, 'Ag': 2}),

    Reaction(['Ag^+', 'Cl^-'], ['AgCl'], {'Ag^+': 1, 'Cl^-': 1, 'AgCl': 1}),
    Reaction(['Ba^2+', 'SO4^2-'], ['BaSO4'], {'Ba^2+': 1, 'SO4^2-': 1, 'BaSO4': 1}),
    Reaction(['Fe^3+', 'SCN^-'], ['Fe(SCN)3'], {'Fe^3+': 1, 'SCN^-': 3, 'Fe(SCN)3': 1}),
    Reaction(['Fe^3+', 'OH^-'], ['Fe(OH)3'], {'Fe^3+': 1, 'OH^-': 3, 'Fe(OH)3': 1}),
    Reaction(['Cu^2+', 'OH^-'], ['Cu(OH)2'], {'Cu^2+': 1, 'OH^-': 2, 'Cu(OH)2': 1}),
    Reaction(['Zn^2+', 'OH^-'], ['Zn(OH)2'], {'Zn^2+': 1, 'OH^-': 2, 'Zn(OH)2': 1}),
    Reaction(['Fe^2+', 'OH^-'], ['Fe(OH)2'], {'Fe^2+': 1, 'OH^-': 2, 'Fe(OH)2': 1}),
    Reaction(['Co^2+', 'OH^-'], ['Co(OH)2'], {'Co^2+': 1, 'OH^-': 2, 'Co(OH)2': 1}),
    Reaction(['Co^2+', 'Cl^-'], ['[CoCl4]^2-'], {'Co^2+': 1, 'Cl^-': 4, '[CoCl4]^2-': 1}),
    Reaction(['Mn^2+', 'OH^-'], ['Mn(OH)2'], {'Mn^2+': 1, 'OH^-': 2, 'Mn(OH)2': 1}),
    Reaction(['Ca^2+', 'OH^-'], ['Ca(OH)2'], {'Ca^2+': 1, 'OH^-': 2, 'Ca(OH)2': 1}),
    Reaction(['Sn^4+', 'OH^-'], ['SnO2', 'H2O'], {'Sn^4+': 1, 'OH^-': 4, 'SnO2': 1, 'H2O': 2}),
    Reaction(['Ag^+', 'OH^-'], ['Ag2O', 'H2O'], {'Ag^+': 2, 'OH^-': 2, 'Ag2O': 1, 'H2O': 1}),
    Reaction(['Al^3+', 'OH^-'], ['Al(OH)3'], {'Al^3+': 1, 'OH^-': 3, 'Al(OH)3': 1}),
    Reaction(['Al(OH)3', 'OH^-'], ['Al(OH)4^-'], {'Al(OH)3': 1, 'OH^-': 3, 'Al(OH)4^-': 1}),
    Reaction(['Pb^2+', 'Br^-'], ['[PbBr4]^2-'], {'Pb^2+': 1, 'Br^-': 4, '[PbBr4]^2-': 1}),
    Reaction(['H^+', 'SO3^2-'], ['HSO3^-'], {'H^+': 1, 'SO3^2-': 1, 'HSO3^-': 1}),
]

reaction_db = ReactionDatabase(reactions)

# Print reactants without enthalpy information
print("no_enthalpy_information:", list(set(no_enthalpy)), '\n')