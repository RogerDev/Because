# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------
""" This test is useful for exploring the limits of sensitivity using various
    distributions.  If the current pattern is followed, the signal to noise
    ratio can be set for each distribution type.
    It is important to name paired items with the same unique prefix
    (e.g. Norm1, Norm2; NormLogis1, NormLogis2).
    We want to test both ends of the SNR range (e.g. SNR=5, and SNR=1/5)
"""
# Initialize any variables here
t = 0
SNR = 1 # Signal to Noise Ratio

# Describe the test
testDescript = 'Pairwise test for testing sensitivity and directionality'

# Define the causal model.
# Each random variable has the following fields:
# - Name
# - List of Parents
# - isObserved (Optional, default True)
# - Data Type (Optional, default 'Numeric')  
model =     [('Norm1', []),
			('Norm2' , ['Norm1']),
            ('Logis1', []),
			('Logis2' , ['Logis1']),
            ('Exp1', []),
			('Exp2' , ['Exp1']),
			]

# Structural Equation Model (SEM)
varEquations = [
			    'Norm1 = normal(-1, 1)',
			    'Norm2 = Norm1 + normal(0,1)',
			    'Logis1 = logistic(1,SNR)',
                'Logis2 = Logis1 + logistic(0,1)',
                'Exp1 = exponential() * SNR',
                'Exp2 = Exp1 + exponential() * 1'
		        ]


				
