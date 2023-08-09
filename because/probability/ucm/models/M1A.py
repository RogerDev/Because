# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------
from probability.ucm.ucm import UCVar
from random import seed
seed(1)

# Initialize any variables here
# For models variables, initialize the variables outside the model equations
varA = UCVar([0.9, 0.1])
varC = UCVar([0.2, 0.4, 0.4])
varB = UCVar([0.2, 0.1, 0.1, 0.1, 0.5], parent=[varA, varC])

# print("\nA Distribution: \t")
# pprint.pprint(varA.distribution)
# print("\nC Distribution: \t")
# pprint.pprint(varC.distribution)
# print("\nB Distribution: \t")
# pprint.pprint(varB.distribution)


# Describe the test
testDescript = 'Reference Model M1A Categorical -- simple V structure'

# Define the causal model.
# Each random variable has the following fields:
# - Name
# - List of Parents
# - isObserved (Optional, default True)
# - Data Type (Optional, default 'Numeric')
model = [('A', []),
         ('C', []),
         ('B', ['A', 'C'])
         ]

# Structural Equation Model for data generation
varEquations = [
    'A = varA.get_value()',
    'C = varC.get_value()',
    'B = varB.get_value([A,C])'
]