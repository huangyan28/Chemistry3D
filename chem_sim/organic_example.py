from simulation.simulator import Container

# Initialization of Reagent Bottles
# The ‘solute’ parameter is a dictionary that contains the solute's name and amount.
# The ‘volume’ parameter specifies the total volume of the solution.
# The ‘org’ parameter distinguishes between organic and inorganic reactions.

a1 = Container({'BrBr': 20}, org=True, volume=10)
a2 = Container({'c1ccc2cc3ccccc3cc2c1': 10}, org=True, volume=10)

# Initialization of Sampling Bottles
b1 = Container(org=True)
b2 = Container(org=True)

# sample
mid1 = b1.update(a1, 5)
b2.update(a2)

# reaction
mid2 = b1.update(b2)
info = b1.get_info(verbose=True)

print('mid1:', mid1)
print('mid2:')
for i in mid2[10:]:
    print(i)
