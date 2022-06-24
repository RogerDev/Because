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
			'D',
			'E',
			'C',
			'A'
			]

# Structural Equation Model for data generation
varEquations = [
                'B = logistic(-2,1) if choice([0,1]) else logistic(3,.75)',
                'C = normal(1,3) if choice([0,1]) else abs(normal(-1,1))',
                'D = normal(-3,1) if choice([0,1]) else abs(normal(2,1))',
                'E = exponential() * 4',
			    #'A = math.tanh(B) + math.sin(B)  + logistic(0,.3)',
			    #'A = (math.tanh(B) if choice([0,1]) else math.sin(B))  + logistic(0,.3)',
                #'A = math.tanh(B) + logistic(0, .3)',
                'A = tanh(B) + sin(C*2) + cos(D*1.1) + tanh(E) + logistic(0,.3)',
				#'A = -3 * B + 2 * C',
                #'A = B + logistic(0,.3)',
			    #'C =  B + normal(0,2)',

		        ]
