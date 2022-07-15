# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------

# Data generator for testing the probability module (prob.py).
# Do not modify this file unless making corresponding changes to probTest.py
# Variables A - C are a deterministic emulation of two dice being rolled (ala Craps).
# A is die 1
# B is die 2
# C is the total roll:  die1 + die2
# N is a standardized normal distribution
# N2 is N + an offset normal distribution with mean = 1

# Initialize any variables here
t = 0
bRange = 2
# Describe the test
testDescript = 'Reference Data for probability testing.  Do not modify this file unless making corresponding changes to probTest.py'

# Define the causal model.
# Each random variable has the following fields:
# - Name
# - List of Parents
# - isObserved (Optional, default True)
# - Data Type (Optional, default 'Numeric')
model =    [
            # N and N2 are Gaussians in relation N -> N2
            ('N', []),
            ('N2', ['N']),
            ('N3', ['N']),
            ('N4', ['N']),
            ('N5', ['N']),
            ('N6', ['N']),
            ('N7', ['N']),
            ('N8', ['N']),
            ('N9', ['N']),
            ('N10', ['N'])
			]

# Structural Equation Model for data generation
varEquations = [
                'N = logistic(0,1)',
                'N2 = math.tanh(N) + logistic(0,0.1)',
                'N3 = math.tanh(N) + logistic(0,0.2)',
                'N4 = math.tanh(N) + logistic(0,0.5)',
                'N5 = math.tanh(N) + logistic(0,1)',
                'N6 = math.tanh(N) + logistic(0,1.5)',
                'N7 = math.tanh(N) + logistic(0,2)',
                'N8 = math.tanh(N) + logistic(0,3)',
                'N9 = math.tanh(N) + logistic(0,5)',
                'N10 = math.tanh(N) + logistic(0,10)'
		        ]
