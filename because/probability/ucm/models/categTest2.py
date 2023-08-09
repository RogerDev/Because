# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------
from itertools import permutations
from numpy.random import choice, seed

# Initialize any variables here
seed(1)
genderDist = [0.4, 0.6]
sizes = ["xs", "s", "m", "l", "xl"]
base_dist = [0.05, 0.1, 0.15, 0.3, 0.4]
dist_perm = list(permutations(base_dist))
sizeProb = {
    'm': dist_perm[choice(range(len(dist_perm)))],
    'f': dist_perm[choice(range(len(dist_perm)))]
}

#sizeProb = {
#     'm': [0.05, 0.15, 0.4, 0.3, 0.1],
#     'f': [0.1, 0.3, 0.4, 0.15, 0.05]
# }

# Describe the test
testDescript = 'Test with 2 trinary models variables'

# Define the causal model.
# Each random variable has the following fields:
# - Name
# - List of Parents
# - isObserved (Optional, default True)
# - Data Type (Optional, default 'Numeric')
# model = [('X', [], 'Categorical'),
#          ('Y', ['X'], 'Categorical)
#          ]

# varEquations = [
# 'X = choice(range(1,4), p=[0.25,0.25,0.5])',
# 'Y = X if uniform(0,1) > 0.9 else choice(range(1,4))'
# ]

# Gender Equations
model = [('gender', []),
         ('size', ['gender'])
         ]
varEquations = [
    'gender = choice(["m", "f"], p=genderDist)',
    'size = choice(sizes, p=sizeProb[gender])'
]
