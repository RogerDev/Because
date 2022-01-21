# ----------------------------------------------------------------------------
# Model Definitions
#

# Initialize any variables here
t = 0

# Describe the test
testDescript = 'Reference Model M4'

# Define the causal model.
# Each random variable has the following fields:
# - Name
# - List of Parents
# - isObserved (Optional, default True)
# - Data Type (Optional, default 'Numeric')
model =    ['A',
            'B',
            'C',
            'A2',
            'AB1',
            'AB2',
            'AB3',
            'AB4',
            'B2',
            'BC1',
            'ABC1',
            ]

# Structural Equation Model for data generation
varEquations = [
                'A = normal(0,1) + logistic(1, .5)',
                'B = logistic(0,1)',
                'C = logistic(-2, 1)',
                'A2 = A + logistic(0,.5)',
                'AB1 = A + B + logistic(0,.5)',
                'AB2 = AB1 + uniform(.5, 1.5)',
                'AB3 = A2 + AB2 + normal(0,.5)',
                'AB4 = AB3 + normal(0,.5)',
                'B2 = B + logistic(.5, .3)',
                'BC1 = B2 + C + logistic(0, .4)',
                'ABC1 = AB3 + BC1 + logistic(0,.5)',
		        ]
				
