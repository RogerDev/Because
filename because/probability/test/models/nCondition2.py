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
model =    ['A2', 'A3', 'A4', 'A5', 'A6','B','C', 'D', 'E', 'F', 'I1', 'I2', 'L1', 'L2', 'L3',
			]

# Structural Equation Model for data generation
varEquations = [
                # Bimodal logistic - logistic mixture
                'B = logistic(-2,1) if choice([0,1]) else logistic(3,.75)',
                # Bimodal normal - logistic mixture
                'C = normal(1.5,1) if choice([0,1,1]) else logistic(-.5, 2)',
                # Bimodal logistic - normal mixture
                'D = truncated("logistic(1,1) if choice([0,0, 1]) else normal(1.2,.75)", -2, 4)',
                #'D = logistic(1,1) if choice([0,0, 1]) else normal(1.2,.75)',
                # Half-normal distribution
                'E = truncated("abs(normal(1,1))", None,2)',
                # Exponential distribution
                'F = exponential() * .5',
                'f1 = math.tanh(B+1)',
                'f2 = math.sin(C*0.75)',
                'f3 = math.tanh(D-2)',
                'f4 = math.cos(E*1.2)',
                'f5 = math.tanh(F+3)',
                'cn = [f1, f2, f3, f4, f5]',
                'A2 = 0 if sum(cn[:1]) < -0.5 else 1 if -0.5 <= sum(cn[:1]) < 0 else 2 if 0 <= sum(cn[:1]) < 0.5 else 3',
                'A3 = 0 if sum(cn[:2]) < -0.5 else 1 if -0.5 <= sum(cn[:2]) < 0 else 2 if 0 <= sum(cn[:2]) < 0.5 else 3',
                'A4 = 0 if sum(cn[:3]) < -0.5 else 1 if -0.5 <= sum(cn[:3]) < 0 else 2 if 0 <= sum(cn[:3]) < 0.5 else 3',
                'A5 = 0 if sum(cn[:4]) < -0.5 else 1 if -0.5 <= sum(cn[:4]) < 0 else 2 if 0 <= sum(cn[:4]) < 0.5 else 3',
                'A6 = 0 if sum(cn[:5]) < -0.5 else 1 if -0.5 <= sum(cn[:5]) < 0 else 2 if 0 <= sum(cn[:5]) < 0.5 else 3',
                'I1 = normal(0,1)',
                'I2 = normal(0,1)',
                'L1 = 1.2 * B + logistic(2, max([(B+5)/10, .1]))',
                'L2 =  1.2 * B + f2 + logistic(-2, .5)',
                'L3 = -2 * C  + 2 * B + normal(2, 1.2)',
		        ]
