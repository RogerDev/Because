# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------

# Initialize any variables here
bDist = [0.3, 0.7]
aDist = {
    0: [0.2, 0.8],
    1: [0.8, 0.2]
}

# Non UC distribution
cDist = {
    0: [0.1, 0.9],
    1: [0.4, 0.6]
}
## Uniform channel distribution
# cDist = {
#     0: [0.1, 0.9],
#     1: [0.9, 0.1]
# }


# Describe the test
testDescript = 'Reference Model M1B Categorical -- Simple Inverted V'

# Define the causal model.
# Each random variable has the following fields:
# - Name
# - List of Parents
# - isObserved (Optional, default True)
# - Data Type (Optional, default 'Numeric')
model =    [('B', []),
            ('A', ['B']),
            ('C', ['B']),
            ]

varEquations = [
    'B = choice(range(0,2), p=bDist)',
    'A = choice(range(0,2), p=aDist[B])',
    'C = choice(range(0,2), p=cDist[B])',
]



