# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------

# Initialize any variables here
t = 0

# Describe the test
testDescript = 'Reference Model M1B-- Simple Inveted V'

# Define the causal model.
# Each random variable has the following fields:
# - Name
# - List of Parents
# - isObserved (Optional, default True)
# - Data Type (Optional, default 'Numeric')  
model =    [('B', []),
			('A', ['B']),
			('C', ['B']),
			]

varEquations = [
			    'B = logistic(0,1)',
			    'A = B + logistic(0,.5)',
			    'C = B  + logistic(0, .5)',
		        ]


				
