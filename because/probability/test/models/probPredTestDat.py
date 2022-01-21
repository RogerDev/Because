# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------

# Data generator for testing the prediction capability of probability module (prob.py).
#

# Initialize any variables here
import math
t = 0
bRange = 2
# Describe the test
testDescript = 'Reference Data for probability prediction testing.  Do not modify this file unless making corresponding changes to probPredTest.py'

# Define the causal model.
# Each random variable has the following fields:
# - Name
# - List of Parents
# - isObserved (Optional, default True)
# - Data Type (Optional, default 'Numeric')
model =    [
            # A, B, and C are Dice (A + B) and the total roll (C)
            ('X1', []),
            ('X2', []),
            ('X3', []),
            ('Y', ['X1', 'X2', 'X3', 'X4']),
            ('DX1', []),
            ('DX2', []),
            ('DX3', []),
            ('DX4', []),
            ('DY', ['DX1', 'DX2', 'DX3', 'DX4']),
			]

# Structural Equation Model for data generation
varEquations = [
                # Continuous variables (X's and Y are for 'training', XTs and YT are for testing)
                'X1 = normal(0,1)',
                'X2 = logistic(2,1)',
                'X3 = exponential()',
                'Y = X1 + abs(X2)**.7 + math.log(X3)',
                # Discrete varibles
                'DX1 = choice(range(1, 7))',
                'DX2 = choice(range(-1, 2))',
                'DX3 = choice(range(-2, 11))',
                'DX4 = 1 if normal(0,1) > .7 else 0',
                'DY = DX1 + DX2 if DX4 > 0 else DX3 - DX2 + DX1',
		        ]
