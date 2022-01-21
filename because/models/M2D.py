# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------

# Initialize any variables here
t = 0

# Describe the test
testDescript = 'Reference Model M2'

# Define the causal model.
# Each random variable has the following fields:
# - Name
# - List of Parents
# - isObserved (Optional, default True)
# - Data Type (Optional, default 'Numeric')
model =    [('B', []),
			('A' , ['B']),
			('C', ['B', 'A']),
			] 

# Structural Equation Model for data generation
varEquations = [
			    'B = 0 if uniform(0, 1) < .75 else 1',
                #'A =  0 if (B == 0 and uniform(0,1) < .8) or (B == 1 and uniform(0,1) < .2) else 1',
                #'C = A + B if uniform(0,1) < .5 else A+2 * B if uniform(0,1) < .3 else 0',
                #'C = A + 2 * B',
			    'A = normal(1, .5) if B == 0 else normal(1.5, 1)',
			    'C = A * 2 + normal(.5, 1) if B == 0 else A * 2 + normal(1, 2)',
                #'C = normal(0,.5) if B == 0 else normal(.5,.5)',
                #'C = A * 1.5',
                't = t + 1'
		        ]
