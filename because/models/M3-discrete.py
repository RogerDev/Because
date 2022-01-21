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
model =    [('B', []),
			('A' , ['B']),
            ('D', ['A']),
			('C', ['B', 'D']),
			]

# Structural Equation Model for data generation
varEquations = [
                #'B = choice(range(-bRange, bRange+1))',
                'A = choice(range(1, 7))',
			    #'B = 2 * B if uniform() < .2 else B',
                'B = choice(range(1, 7))',
                #'D = .5 * A + logistic(3,1)',
			    #'C = .5 * B + 1 * D + logistic(0,1)',
                'C = A + B',
                't = t + 1'
		        ]
