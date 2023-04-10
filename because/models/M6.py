# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------

# Initialize any variables here
t = 0

# Describe the test
testDescript = 'Reference Model M6'

# Define the causal model.
# Each random variable has the following fields:
# - Name
# - List of Parents
# - isObserved (Optional, default True)
# - Data Type (Optional, default 'Numeric')
model = [   ('B', []),
			('A' , ['B']),
			('C', ['A', 'B', 'D']),
            ('D', ['A']),
            ('E', ['C'])
		]

# Structural Equation Model for data generation
varEquations = [
			    'B = logistic(0,1)',
			    'A = 1 * B + logistic(0,.5)',
                'D = 1 * A + logistic(0,.5)',
			    'C = .33 * A + .33 * B + .33 * D + logistic(0,.5)',
                'E = 1 * C + logistic(0,.5)',
		        ]
