# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------

# Initialize any variables here
t = 0

# Describe the test
testDescript = 'Reference Model M7'

# Define the causal model.
# Each random variable has the following fields:
# - Name
# - List of Parents
# - isObserved (Optional, default True)
# - Data Type (Optional, default 'Numeric')  
model = [   ('B', []),
            ('F', [], True),
            ('G', []),
			('A' , ['B', 'F']),
			('C', ['B', 'D']),
            ('D', ['A', 'G']),
            ('E', ['C'])
		]

varEquations = [
			    #'B = math.sin((t % 365) / 365 * 6.28) * 50 + 40 + normal(0, 5)',
                'B = logistic(0, 1)',
                'F = logistic(-1, 1)',
                'G = logistic(1, 1)',
			    'A = .5 * B + .5 * F + logistic(0, .5)',
                'D = .5 * A + .5 * G + logistic(0, .5)',
 			    'C = .5 * B + .5 * D + logistic(0, .5)',
                'E = C + logistic(0,.5)',
		        ]
