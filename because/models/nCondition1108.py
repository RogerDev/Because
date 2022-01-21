# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------

# Initialize any variables here
t = 0
from random import *


# Describe the test
testDescript = 'N Common Causes Test'

# Define the causal model.
# Each random variable has the following fields:
# - Name
# - List of Parents
# - isObserved (Optional, default True)
# - Data Type (Optional, default 'Numeric')
model =    ['A2', 'A3', 'A4', 'A5', 'A6','B','C', 'D', 'E', 'F'
			]

# Structural Equation Model for data generation
varEquations = [
                'B = logistic(-2,1) if choice([0,1]) else logistic(3,.75)',
                #'B = normal(0,1)',
                'C = normal(2,1)',
                'D = normal(-1,1)',
                'E = normal(1,1)',
                'F = normal(0,1)',
                'f1 = math.tanh(B)',
                #'f1 = abs(B)**1.1',
                'f2 = math.sin(C)',
                #'f2 = -abs(C)**.9',
                'f3 = math.tanh(D)',
                'f4 = math.cos(E)',
                #'f5 = F**2',
                'f5 = math.tanh(F)',
                'cn = [f1, f2, f3, f4, f5]',
                'A2 = sum(cn[:1]) + logistic(0,.3)',
                'A3 = sum(cn[:2]) + logistic(0,.3)',
                'A4 = sum(cn[:3]) + logistic(0,.3)',
                'A5 = sum(cn[:4]) + logistic(0,.3)',
                'A6 = sum(cn[:5]) + logistic(0,.3)',
		        ]
