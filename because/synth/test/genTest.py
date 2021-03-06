"""
Example file to create synthetic samples in-line without input or output files.
See synth/models/example.py for details on sem and model formats.
"""
from because.synth import gen_data

# We're only interested in generating data here, and not with a comparison causal model,
# so mod is just a list of variables.  See example.py for other uses of the causal model.
mod = ['A',
        'B',
        'C',
        ]

# The Structural Equation Model (SEM) from which to generate data.
sem = ['A=normal(0,2)',
        'B=logistic(2,3)',
        'C=A+B+noise()',
      ]

# Constuct Gen with mod and sem params rather than using an input model file.
gen = gen_data.Gen(mod=mod, sem=sem)

# Get the variable names in the same order as the samples will be generated
variables = gen.getVariables()
print('Vars = ', variables)

# Note that samples here is a generator function, so it, in effect, creates a
# stream that is only produced as it is consumed.
samples = gen.samples(10)

for sample in samples:
    print(sample)

# getSEM() returns the fully realized SEM (i.e. with noise() and coef() replaced
# with actual values.
newSem = gen.getSEM()
print('SEM = ', newSem)

# Generate samples again with reset = False.  The resulting SEM should be the same
# as the first.
samples2 = gen.samples(10, reset=False)
s = [x for x in samples2] # Note samples wont get executed unless we iterate through.
newSem2 = gen.getSEM()
#print('SEM2 = ', newSem2)
if newSem != newSem2:
    print('*** ERROR -- SEM and SEM2 should be identical')

# Generate agiain with reset = True.  The resulting SEM should be different.
samples3 = gen.samples(10, reset=True)
s = [x for x in samples3]
newSem3 = gen.getSEM()
#print('SEM3 = ', newSem3)
if newSem == newSem3:
    print('*** ERROR -- SEM and SEM3 should not be identical')

    
