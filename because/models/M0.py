# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------

# Initialize any variables here
t = 0

# Describe the test
testDescript = 'Reference Model M0-- Two variable chain'

# Define the causal model.
# Each random variable has the following fields:
# - Name
# - List of Parents
# - isObserved (Optional, default True)
# - Data Type (Optional, default 'Numeric')  
model =    [('A', []),
			('B', ['A'])
			]

varEquations = [
			    'A = logistic(0,1)',
			    'B = A + logistic(0, .5)',
		        ]


				
