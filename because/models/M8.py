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
            ('F', []),
            ('G', []),
			('A' , ['B', 'F']),
			('C', ['B', 'A', 'D']),
            ('D', ['A', 'G']),
            ('E', ['C'])
		]

varEquations = [
			    #'B = math.sin((t % 365) / 365 * 6.28) * 50 + 40 + normal(0, 5)',
                'B = logistic(0, 1)',
                'F = logistic(0, 1)',
                'G = logistic(0, 1)',
			    'A = (B + F) / 2.0 + logistic(0,.1)',
                'D = (A + G) / 2.0 + logistic(0,.1)',
 			    'C = (B + A + D) / 3.0 + logistic(0,.1)',
                'E = C + logistic(0,3)',
		        ]
