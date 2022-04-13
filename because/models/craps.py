# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------

# Initialize any variables here
t = 0

# Describe the test
testDescript = 'Temporary Craps Test'

# Define the causal model.
# Each random variable has the following fields:
# - Name
# - List of Parents
# - isObserved (Optional, default True)
# - Data Type (Optional, default 'Numeric')
model =    ['D1',
            'D2',
            'ROLL',
			] 

# Structural Equation Model for data generation
varEquations = [
			    'D1 = choice(range(1,7))',
                'D2 = choice(range(1,7))',
                'ROLL = D1 + D2',
 		        ]
