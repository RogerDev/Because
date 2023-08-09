# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------

# Initialize any variables here
t = 0

# Describe the test
testDescript = 'Test with binary models values '

# Define the causal model.
# Each random variable has the following fields:
# - Name
# - List of Parents
# - isObserved (Optional, default True)
# - Data Type (Optional, default 'Numeric')
model =    [('X', []),
            ('Y', ['X'])
            ]

# varEquations = [
#     'X = 1 if uniform(0,1) > 0.7 else 0',
#     'U = uniform(0,1)',
#     # 'Y = 0 if U < 0.4 and X == 0 else 1 if U >= 0.4 and X == 0 else 0 if U > 0.1 and X == 1 else 1 if U <= 0.1 and X==1 else 0'
#     # 'Y = 0 if U < 0.5 and X == 1 else 1 if 0.5 < U < 0.9 and X == 1 else 0 if U < 0.2 and X == 0 else 1 if 0.2 < U < 0.4 and X == 0 else 2'
#     'Y = 1^X if U > 0.1 else 0^X'
# ]

varEquations = [
    'X = 1 if uniform(0,1) > 0.7 else 0',
    'U = uniform(0,1)',
    'Y = 1^X if U > 0.1 else 0^X'
]
