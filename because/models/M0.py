# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------

# Initialize any variables here
t = 0

# Describe the test
testDescript = 'Reference Model M0-- Simple V'

# Define the causal model.
# Each random variable has the following fields:
# - Name
# - List of Parents
# - isObserved (Optional, default True)
# - Data Type (Optional, default 'Numeric')  
model =    [('A', []),
			('C' , []),
			('B', ['A', 'C'])
			]

varEquations = [
			    'A = logistic(5,3)',
			    'C = logistic(0,3)',
			    'B = A + C + logistic(0, 6)',
		        ]


				
