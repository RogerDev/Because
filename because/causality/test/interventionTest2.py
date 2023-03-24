"""
Implement an "IceCream <- Temperature -> Crime" model
Then:
- Assess the apparent effect of IceCream on Crime
- Use several different methods to detect the causal effect
    - Explicit Controlling for Temperature
    - Use Expected value queries with intervention (i.e. do())
    - Use distribution queries with intervention and take expected
        value of the distibutions.
    - Use ACE() to compute.
"""
import sys

from because.causality import RV
from because.causality import cGraph
from because.synth import gen_data
from because.probability import prob
from because.probability import independence
from because.causality import cquery
cquery = cquery.query
prob.DEBUG = False

# Number of samples
nSamples = 10000

# Power to use for queries
pwr = 1

# Causal effect we should observe (i.e. ce) should be left at zero
SEM = [ 
    'ce = 0.0', # Causal effect we want to detect
    'Day = uniform(0, 364)',
    'Temperature = sin(Day / (2*pi)) * 30 + 40 + normal(0,5)',
    'IceCream = 1000 + 10 * Temperature + normal(0,300)',
    'Crime = ce * IceCream + 50 + 3 * Temperature + normal(0, 2)'
    ]

# Constuct Gen with mod and sem params rather than using an input model file.
gen = gen_data.Gen(mod=['Temperature', 'IceCream', 'Crime'], sem=SEM)

# Get the variable names in the same order as the samples will be generated
variables = gen.getVariables()
print('Vars = ', variables)

# Note that samples here is a generator function, so it, in effect, creates a
# stream that is only produced as it is consumed.
data = gen.getDataset(nSamples)

# Assume that there is a causal effect of IceCream on Crime
model = [
        ('Temperature', []),
        ('IceCream', ['Temperature']),
        ('Crime', ['Temperature', 'IceCream'])
]
gnodes = []
# 'model' is set when the text file is exec'ed
for var in model:
    observed = True
    dType = 'Numeric'
    name, parents = var[:2]
    if len(var) >= 3:
        observed = var[2]
    if len(var) >= 4:
        dType = var[3]
    gnode = RV(name, parents, observed, dType, None, None)
    gnodes.append(gnode)

# Causal Graph
g = cGraph(gnodes, data, power=pwr)

verbosity = 0

# Pick a high and low value for IceCream so we can see its
# effect on Crime
iHigh = 1700
iLow = 1300

print('Apparent Effect.  Should be non zero:')
h1 = cquery(g, 'E(Crime | IceCream=' + str(iHigh) + ')', verbosity=verbosity, power=pwr)
l1 = cquery(g, 'E(Crime | IceCream=' + str(iLow) + ')', verbosity=verbosity, power=pwr)
print('  E(Crime | IceCream = iHigh) = ', h1)
print('  E(Crime | IceCream = iLow) = ', l1)
print('  Diff =', h1 - l1)
print('  Apparent effect = ', (h1-l1) / (iHigh - iLow))

print()
print('Use explicit controllng for temperature:')
h2 = cquery(g, 'E(Crime | IceCream=' + str(iHigh) + ', Temperature)', verbosity=verbosity, power=pwr)
l2 = cquery(g, 'E(Crime | IceCream=' + str(iLow) + ', Temperature)', verbosity=verbosity, power=pwr)
print('  E(Crime | IceCream = iHigh, Temperature) = ', h2)
print('  E(Crime | IceCream = iLow, Temperature) = ', l2)
print('  Diff = ', h2 - l2)
print('  Implied ACE = ', (h2-l2)/ (iHigh - iLow))

print()
print('Use Expected value intervention:')
h3 = cquery(g, 'E(Crime | do(IceCream={val}))'.format(val=iHigh), verbosity=verbosity, power=pwr)
l3 = cquery(g, 'E(Crime | do(IceCream={val}))'.format(val=iLow), verbosity=verbosity, power=pwr)
print('  E(Crime | do(IceCream=iHigh)) = ', h3)
print('  E(C | do(IceCream=iLow)) = ', l3)
print('  Diff = ', h3 - l3)
print('  Implied ACE = ', (h3-l3)/ (iHigh - iLow))

print()
print('Use distribution intervention:')
dh4 = cquery(g, 'P(Crime | do(IceCream={val}))'.format(val=iHigh), verbosity=verbosity, power=pwr)
dl4 = cquery(g, 'P(Crime | do(IceCream={val}))'.format(val=iLow), verbosity=verbosity, power=pwr)
h4 = dh4.E()
l4 = dl4.E()
print('  P(Crime | do(IceCream=iHigh)).E() = ', h4)
print('  P(C | do(IceCream=iLow)).E() = ', l4)
print('  Diff = ', h4 - l4)
print('  Implied ACE = ', (h4-l4)/ (iHigh - iLow))

print()
print('Use ACE Metric:')
ace = g.ACE('IceCream', 'Crime', power=pwr)
print('  ACE(IceCream, Crime) = ', ace)

mde = g.MDE('IceCream', 'Crime')

#print()
#print('MDE = ', mde)