import re

def mix_color(rgba1, rgba2):
    """
    Mixes two RGBA colors and returns the resulting color.

    Args:
        rgba1 (list): The first color as [r, g, b, a] where r, g, b are in the range [0, 255] and a is in the range [0, 1].
        rgba2 (list): The second color as [r, g, b, a] where r, g, b are in the range [0, 255] and a is in the range [0, 1].

    Returns:
        list: The resulting mixed color as [r, g, b, a].
    """
    r1, g1, b1, a1 = rgba1
    r2, g2, b2, a2 = rgba2

    # Calculate the total alpha value
    alpha_total = 1 - (1 - a2) * (1 - a1)

    # Calculate the resulting color channels
    r = (r1 * a1 + r2 * a2 * (1 - a1)) / alpha_total
    g = (g1 * a1 + g2 * a2 * (1 - a1)) / alpha_total
    b = (b1 * a1 + b2 * a2 * (1 - a1)) / alpha_total

    return [r, g, b, alpha_total]


def charge_balance(input_reactants, input_reactant_amounts):
    """
    Calculates the total charge of a set of reactants.

    Args:
        input_reactants (list): List of reactant formulas as strings.
        input_reactant_amounts (list): List of amounts of each reactant.

    Returns:
        int: The total charge of the reactants.
    """
    total_charge = 0

    for reactant, amounts in zip(input_reactants, input_reactant_amounts):
        charge = 0

        # Check if the reactant has a charge symbol
        if '^' in reactant:
            sign_index = reactant.index('^')
            charge_str = reactant[sign_index + 1:]
            if charge_str.startswith('+'):
                charge = amounts
            elif charge_str.startswith('-'):
                charge = -amounts
            else:
                charge = int(charge_str[::-1]) * amounts

        total_charge += charge

    return total_charge


def ionization_constant(temperature):
    """
    Calculates the ionization constant of water (Kw) at a given temperature using linear interpolation.

    Args:
        temperature (float): The temperature in degrees Celsius.

    Returns:
        float: The ionization constant (Kw) at the given temperature.
    """
    temperature_table = [0, 10, 20, 25, 40, 50, 90, 100]
    kw_table = [0.114, 0.292, 0.681, 1.01, 2.92, 5.47, 38.0, 55.0]

    if temperature in temperature_table:
        index = temperature_table.index(temperature)
        return kw_table[index]

    # Find the closest temperatures for interpolation
    lower_temp = max([temp for temp in temperature_table if temp <= temperature] + [min(temperature_table)])
    upper_temp = min([temp for temp in temperature_table if temp >= temperature] + [max(temperature_table)])

    # Get the corresponding Kw values
    lower_index = temperature_table.index(lower_temp)
    upper_index = temperature_table.index(upper_temp)
    lower_kw = kw_table[lower_index]
    upper_kw = kw_table[upper_index]

    # Linear interpolation to calculate Kw at the given temperature
    if upper_index == 0:
        return 1
    elif lower_index == 100:
        return 1
    else:
        ionization_constant = lower_kw + ((temperature - lower_temp) / (upper_temp - lower_temp)) * (upper_kw - lower_kw)

    return ionization_constant


def temperature(volume, heat_input, initial_temperature):
    """
    Calculates the final temperature of water given its volume, heat input, and initial temperature.

    Args:
        volume (float): The volume of water in liters.
        heat_input (float): The heat input in joules.
        initial_temperature (float): The initial temperature in degrees Celsius.

    Returns:
        float: The final temperature in degrees Celsius.
    """
    specific_heat = 4.18  # Specific heat capacity of water in J/g°C
    delta_T = heat_input / (volume * specific_heat * 1000)  # Convert volume to grams and calculate temperature change
    final_temperature = initial_temperature + delta_T  # Calculate final temperature
    return final_temperature
