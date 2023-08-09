# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------

# Initialize any variables here
from probability.ucm.ucm import UCVar
# For models variables, initialize the variables outside the model equations
varB = UCVar([0.9, 0.1])
varA = UCVar([0.25, 0.25, 0.5], parent=varB)
varC = UCVar([0.2, 0.1, 0.1, 0.1, 0.5], parent=[varA, varB])

## Alternative distribution
# varB = UCVar([0.4, 0.5, 0.1])
# varA = UCVar([0.25, 0.25, 0.5], parent=varB)
# varC = UCVar([0.3, 0.7], parent=[varA, varB])

## Alternative distribution
# varB = UCVar([0.2, 0.1, 0.1, 0.1, 0.5])
# varA = UCVar([0.25, 0.25, 0.1, 0.4], parent=varB)
# varC = UCVar([0.2, 0.8], parent=[varA, varB])

# Describe the test
testDescript = 'Reference Model M2'

# Define the causal model.
# Each random variable has the following fields:
# - Name
# - List of Parents
# - isObserved (Optional, default True)
# - Data Type (Optional, default 'Numeric')
model = [('B', []),
         ('A', ['B']),
         ('C', ['B', 'A']),
         ]

# Structural Equation Model for data generation
varEquations = [
    'B = varB.get_value()',
    'A = varA.get_value(B)',
    'C = varC.get_value([A,B])'
]