# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------

# Initialize any variables here
t = 0

# Describe the test
testDescript = 'Reference Model M11'

# Define the causal model.
# Each random variable has the following fields:
# - Name
# - List of Parents
# - isObserved (Optional, default True)
# - Data Type (Optional, default 'Numeric')  
model = [   ('B', []),
            ('F', []),
            ('E', []),
			('A' , ['B', 'F', 'E']),
			('C', ['B', 'F', 'E', 'D']),
            ('D', ['A']),
		]

varEquations = [
			    #'B = math.sin((t % 365) / 365 * 6.28) * 50 + 40 + normal(0, 5)',
                'B = logistic(0, 1)',
                'F = logistic(-1, 1)',
                'G = logistic(-2, 1)',
                'E = logistic(2, 1)',
			    'A = (B + F + E) / 3.0 + logistic(0,.5)',
                'D = A + logistic(0,.5)',
 			    'C = (B + F + E + D) / 4.0 + logistic(0,.5)',
		        ]
