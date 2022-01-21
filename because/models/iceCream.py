# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------

# Initialize any variables here
t = 0

# Describe the test
testDescript = 'Reference Model M1 -- Ice Cream / Crime model'

# Define the causal model.
# Each random variable has the following fields:
# - Name
# - List of Parents
# - isObserved (Optional, default True)
# - Data Type (Optional, default 'Numeric')  
model =    [('temperature', []),
			('iceCream' , ['temperature']),
			('crime', ['temperature'])
			]

varEquations = [
			    'temperature = math.sin((t % 365) / 365 * 6.28) * 50 + 40 + normal(0, 5)',
			    'iceCream = 100000 + 10 * temperature + normal(0, 100)',
			    'crime = 10 + temperature + iceCream * .01 + logistic(0, 10)',
			    #'crime = 10 + temperature + logistic(0, 10)',
                't = t + 1'
		        ]


				
