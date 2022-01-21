# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------

# Initialize any variables here
t = 0
bRange = 2
# Describe the test
testDescript = 'Reference Model M3'

# Define the causal model.
# Each random variable has the following fields:
# - Name
# - List of Parents
# - isObserved (Optional, default True)
# - Data Type (Optional, default 'Numeric')
model =    [
            ('L1', []),
            ('L2', []),
            ('L3', []),
            ('E1', []),
            ('E2', []),
            ('N1', []),
            ('N2', []),
            ('L4', []),
            ('L5', []),
            ('L6', []),
            ('L7', []),
            ('L8', []),
            ('L9', []),
            ('E3', []),
            ('N3', []),
            ('N4', []),
            ('N5', []),
            ('M1', []),
            ('B', []),
			('A' , ['B']),
            ('D', ['A']),
			('C', ['B', 'D']),
			] 
# Structural Equation Model for data generation
varEquations = [
                #'B = choice(range(-bRange, bRange+1))',
                # A bunch of independent variables with different distributions
                'L1= logistic(0, 1)',
                'L2 = logistic(0, 1)',
                'L3 = logistic(100, .5)',
                'E1 = exponential()',
                'E2 = exponential()',
                'N1 = normal(0,1)',
                'N2 = normal(-10, 2)',
                # Some dependent variables on above distributions
                'L4 = L3 + logistic(0,.5)',
                'L5 = abs(L2)**2 + logistic(0,.1)',
                'L6 = .5 * L3 + logistic(0,.1)',
                'L7 = 1 * L3 + logistic(0, .1)',
                'L8 = 1 * L1 + logistic(0, 1)',
                'L9 = 1 * L1 + logistic(0, 1)',
                'E3 = E1 + .5 * E2 + logistic(0,.1)',
                'N3 = N1 + N2 + exponential()',
                'N4 = N1 + normal(0, .1)',
                'N5 = N1 + normal(0, .1)',
                'M1 = N3 + E1 + normal(0,.1)',

                # Model M3 with subtle conditional independencies
                'B = logistic(0,1)',
			    'A = abs(B)**1.2 + logistic(0,.1)',
                'D = 1 * A + logistic(0,.1)',
			    'C = .5 * abs(B)**.5 + .5 * D + logistic(0,.1)',
                't = t + 1'
		        ]
