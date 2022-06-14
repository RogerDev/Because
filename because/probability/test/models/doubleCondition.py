# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------

# Initialize any variables here
t = 0
from random import *
bRange = 2

# Describe the test
testDescript = 'Reference Model M1'

# Define the causal model.
# Each random variable has the following fields:
# - Name
# - List of Parents
# - isObserved (Optional, default True)
# - Data Type (Optional, default 'Numeric')
model =    ['B',
			'A' ,
			'C'
			]

# Structural Equation Model for data generation
varEquations = [
                'B = logistic(-2,1) if choice([0,1]) else logistic(3,.75)',
                'C = normal(0, 2)',
			    #'A = math.tanh(B) + math.sin(B)  + logistic(0,.3)',
			    #'A = (math.tanh(B) if choice([0,1]) else math.sin(B))  + logistic(0,.3)',
                #'A = math.tanh(B) + logistic(0, .3)',
                'A = tanh(B) + math.sin(C) + logistic(0,.3)',
				#'A = -3 * B + 2 * C',
                #'A = B + logistic(0,.3)',
			    #'C =  B + normal(0,2)',

		        ]
