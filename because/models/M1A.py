# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------

# Initialize any variables here
t = 0

# Describe the test
testDescript = 'Reference Model M1A -- Simple V'

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
			    'A = logistic(5,1)',
			    'C = logistic(0,1)',
			    'B = (A + C) / 2  + logistic(0, .5)',
		        ]


				
