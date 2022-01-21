"""
This is an annotated example model definition file.
A model definition file consists of three main parts:
- A textual description (see testDescript below)
- A causal model representing a Directed Acyclic Graph (DAG).
	This is used to document the causal structure of the data.
	It is used for validating.  If you are only interested
	in generating the dataset and not doing causal analysis,
	than this can be simply a list of variable names.
	The order of this list will correspond to the order of the
	variable in the generated data file.
	See 'model' below for details.
- A Structural Equation  Model (SEM).  This is the list of equations
	used to generate the multivariate dataset. Later equations
	can use the results of previous equations.  The equations
	are standard python expressions, and can use any functions
	from the python math library, and any distributions from
	the numpy.random module (see numpy documentation).  See
	'varEquations' below for details.
"""
# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------

# Initialize any variables here
t = 0

# Describe the test
testDescript = 'This is the description of the model'

# Define the causal model.
# Each random variable has the following fields:
# - Name
# - List of Parents
# - isObserved (Optional, default True)
# - Data Type (Optional, default 'Numeric')  
model =    [('A', []), # Variable 'A' has no parents (i.e. is exogenous)
			('C' , []), # Variable 'C' has no parents
			('B', ['A', 'C']), # Variable 'B' is dependent on 'A' and 'C'
			('D', ['B']), # D is a child of B
			]

# Define the SEM used to generate the data
varEquations = [
			    'A = logistic(5,3)', # A is drawn from a logistic dist
									 # with mean = 5 and scale = 3.
			    'C = normal(0,3)',   # C is drawn from a normal dist
									 # with mean = 0 and std = 3.
				# B is computed as a non-linear (sin) function of A and C,
				# with added normal noise.
			    'B = sin(.5 * A + 2.0 * C) + normal(0, .6)',
				# D is a linear function of D, with a random coefficient,
				# and with random amount of unspecified noise.
				# Generally it is better to add your own noise and coefs
				# as in B above, but this is usefule for generating a
				# wide range of tests.  If coef() or noise() is used,
			    # then each run will generate samples from a different
				# multivariate distribution.
				'D = coef() * B + noise()',
		        ]


				
