from simulation.simulator import Container
import copy

# Inorganic
# Initialization of Reagent Bottles: Ion Quantity & Volume
# The ‘solute’ parameter is a dictionary that contains the solute's name and amount.
# The ‘volume’ parameter specifies the total volume of the solution.

a1 = Container(solute={'MnO4^-': 1, 'K^+': 1}, volume=4)
a2 = Container(solute={'H^+': 3, 'Cl^-': 3}, volume=1)
a3 = Container(solute={'FeO': 2}, volume='s')

# Initialization of Sampling Bottles
b1 = Container()
b2 = Container()
b3 = Container()

# sample
mid1 = b1.update(a1)
b2.update(a2)
b3.update(a3)
info_0 = b3.get_info()
print(info_0)

# reacting
mid2 = b3.update(b2)
mid3 = b3.update(b1)
info = b3.get_info(verbose=True)

print('mid1:', mid1)
print('mid2:', mid2)
# print('mid3:')
# for i in mid3:
#     print(i)

