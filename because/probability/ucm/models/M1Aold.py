# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------
import numpy
from random import seed
seed(1)
# Initialize any variables here
aDist = [0.25, 0.75]
cDist = [0.3, 0.7]

bBaseDist = []
bDist = {
    0: [1 / 3, 1 / 3, 1 / 3],
    1: [0.25, 0.25, 0.5]
}

## Alternative variable distribution
# aDist = [0.25, 0.75]
# cDist = [0.1, 0.9]
# bDist = {
#     0: [0.8, 0.1, 0.1],
#     1: [0.1, 0.1, 0.8]
# }


# Describe the test
testDescript = 'Reference Model M1A Categorical -- Simple V'

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

varEquations = [
    'A = choice(range(0,2), p=aDist)',
    'C = choice(range(0,2), p=cDist)',
    'B = choice(range(0,3), p=bDist[(A + C) % 2])'
]
