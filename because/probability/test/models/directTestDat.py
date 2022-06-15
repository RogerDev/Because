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
            # M and M2 are Gaussians in relation N -> N2
            ('M', []),
            ('M2', ['M']),
            ('M3', ['M']),
            ('M4', ['M']),
            # IVs are an inverted V formation for testing conditional dependency
            # of continuous variables.
            ('IVB', []),
            ('IVA', ['IVB']),
            ('IVC', ['IVB']),
            # Exponential distribution
            ('EXP', []),
            ('EXP2', ['EXP'])
			]

# Structural Equation Model for data generation
varEquations = [
                'N = logistic(0,1)',
                'N2 = N + logistic(1,1)',
                'N3 = 1 / N + logistic(1,1)',
                'M = logistic(1,1)',
                'M2 = M + logistic(1,1)',
                'M3 = M ** 2 + logistic(1,1)',
                'M4 = M ** 2 + logistic(3,3)',
                'IVB = logistic(0,1)',
                'IVA = IVB ** 3 + logistic(0, 1)',
                'IVC = abs(IVB) ** 0.5 + logistic(0, 1)',
                'EXP = exponential()',
                'EXP2 = math.tanh(EXP) + exponential()',
                't = t + 1'
		        ]
