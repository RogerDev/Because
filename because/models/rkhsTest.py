# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------

# Initialize any variables here
from numpy.random import logistic
dd = {0:"logistic(-2, .5)", 1: "logistic(2, .5)", 2:"logistic(-.5, .3)"}

# Describe the test
testDescript = 'RKHS Test Data'

# Define the causal model.
# Each random variable has the following fields:
# - Name
# - List of Parents
# - isObserved (Optional, default True)
# - Data Type (Optional, default 'Numeric')  
model =    ['X'
			]

varEquations = [
                    #'X = logistic(0,1) + logistic(1,.5) + logistic(-1, .5)'
                    #'X = logistic(0,1)'
                    #'X = exponential()'
                    #'X = logistic(2,.5) if choice([0,1])  else logistic(-2,.5)'
                    'X = eval(dd[choice([0,1,2])])'
		        ]