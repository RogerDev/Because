# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------

# Initialize any variables here
t = 0
from random import *


# Describe the test
testDescript = 'Simple Test'

# Define the causal model.
# Each random variable has the following fields:
# - Name
# - List of Parents
# - isObserved (Optional, default True)
# - Data Type (Optional, default 'Numeric')
model =    ['A2', 'A3', 'A4', 'A5','B','C', 'D', 'E'
			]

# Structural Equation Model for data generation
varEquations = [
                'B = normal(0,1)',
                'C = normal(0, .5)',
                'D = logistic(0,2)',
                'E = normal(0,2)',
                'f1 = 1 * B',
                'f2 = -2 * C',
                'f3 = 1 * D',
                'f4 = 1 * E',
                'cn = [f1, f2, f3, f4]',
                'A2 = sum(cn[:1]) + logistic(0,.3)',
                'A3 = sum(cn[:2]) / 2 + logistic(0,.3)',
                'A4 = sum(cn[:3]) / 3 + logistic(0,.3)',
                'A5 = sum(cn[:4]) / 4 + logistic(0,.3)',
		        ]
