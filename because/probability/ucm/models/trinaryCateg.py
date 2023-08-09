# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------

# Initialize any variables here
t = 0

# Describe the test
testDescript = 'Test with trinary models variables'

# Define the causal model.
# Each random variable has the following fields:
# - Name
# - List of Parents
# - isObserved (Optional, default True)
# - Data Type (Optional, default 'Numeric')
model = [('X', []),
         ('Y', ['X'])
         ]

varEquations = [
    'U = uniform(0,1)',
    'X = 1',
    'X = 2 if U > 0.25 else X',
    'X = 3 if U > 0.5 else X',
    # 'X = choice(range(1, 4))',
    'Y = X if uniform(0,1) > 0.5 else choice(range(1,4))',
]
