# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------

# Initialize any variables here
t = 0

# Describe the test
testDescript = 'Test with discrete values for stats testing'

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
			    'A = 1 if uniform(0,1) > .5 else 0',
			    'C = 1 if uniform(0,1) > .75 else 0',
			    'B = A + C',
		        ]


				
