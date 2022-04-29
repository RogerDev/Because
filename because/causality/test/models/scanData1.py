# ----------------------------------------------------------------------------
# Model Definitions
#

# Initialize any variables here
t = 0

# Describe the test
testDescript = 'ScanTest Reference Model'

# Define the causal model.
# Each random variable has the following fields:
# - Name
# - List of Parents
# - isObserved (Optional, default True)
# - Data Type (Optional, default 'Numeric')
model =    [
            ('C', []),
            ('A', []),
            ('B', []),
            ('A2', ['A']),
            ('AB1', ['A', 'B']),
            ('AB2', ['AB1']),
            ('AB3', ['A2', 'AB2']),
            ('AB4', ['AB3']),
            ('B2', ['B']),
            ('BC1', ['B2', 'C']),
            ('ABC1', ['AB3', 'BC1'])
            ]

# Structural Equation Model for data generation
varEquations = [
                'A = logistic(0,1)',
                'B = logistic(0,1)',
                'C = logistic(0,1)',
                'A2 = A + logistic(0,.5)',
                'AB1 = (A + B) / 2 + logistic(0,.5)',
                'AB2 = AB1 + logistic(0,.5)',
                'AB3 = (A2 + AB2) / 2 + logistic(0, .5)',
                'AB4 = AB3 + logistic(0, .5)',
                'B2 = B + logistic(0, .5)',
                'BC1 = (B2 + C) / 2 + logistic(0,.5)',
                'ABC1 = (AB3 + BC1) / 2 + logistic(0,.5)',
		        ]
				
