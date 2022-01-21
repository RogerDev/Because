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
            # A, B, and C are Dice (A + B) and the total roll (C)
            ('A', []),
			('B' , []),
			('C', ['B', 'A']),
            # N and N2 are Gaussians in relation N -> N2
            ('N', []),
            ('N2', ['N']),
            # IVs are an inverted V formation for testing conditional dependency
            # of continuous variables.
            ('IVA', ['IVB']),
            ('IVB', []),
            ('IVC', ['IVB', 'IVA']),
            # Exponential distribution
            ('EXP', [])
			]

# Structural Equation Model for data generation
varEquations = [
                'A = choice(range(1, 7))',
                'B = choice(range(1, 7))',
                'C = A + B',
                'N = normal(0,1)',
                'N2 = N + normal(1,1)',
                'IVB = logistic(0,1)',
                'IVA = IVB + logistic(0, .1)',
                'IVC = 0 * IVA + IVB + logistic(0, .1)',
                'EXP = exponential()',
                't = t + 1'
		        ]
