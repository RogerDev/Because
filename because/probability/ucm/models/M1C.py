# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------

# Initialize any variables here
bDist = {
    0: [0.6, 0.2, 0.2],
    1: [0.2, 0.2, 0.6]
}

cDist = {
    0: [0.1, 0.1, 0.1, 0.7],
    1: [0.1, 0.7, 0.1, 0.1],
    2: [0.1, 0.1, 0.7, 0.1]
}

# Describe the test
testDescript = 'Reference Model M1C Categorical-- 3 variable chain (growing cardinality) '

# Define the causal model.
# Each random variable has the following fields:
# - Name
# - List of Parents
# - isObserved (Optional, default True)
# - Data Type (Optional, default 'Numeric')
model = [('A', []),
         ('B', ['A']),
         ('C', ['B']),
         ]

varEquations = [
    'A = choice(range(0,2), p=[0.3, 0.7])',
    'B = choice(range(0,3), p=bDist[A])',
    'C = choice(range(0,4), p=cDist[B])',
]
