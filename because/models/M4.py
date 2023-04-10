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
model =    [('B', []),
			('A' , ['B']),
			('C', ['B', 'A', 'D']),
            ('D', ['A'])
			]

# Structural Equation Model for data generation
varEquations = [
			    #'B = math.sin((t % 365) / 365 * 6.28) * 50 + 40 + logistic(0, 5)',
                'B = logistic(0,1)',
			    'A = 1 * B + logistic(0, .5)',
                'D = 1 * A + logistic(0, .5)',
			    'C = .33 * A + .33 * B + .33 * D + logistic(0, .5)',
                't = t + 1'
		        ]
				
