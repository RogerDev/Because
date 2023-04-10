import sys
if '.' not in sys.path:
    sys.path.append('.')
import time


from because.causality import rv
from because.causality.cgraph import cGraph
from because.synth import gen_data
from because.probability import independence

args = sys.argv
if len(args) > 1:
    test = args[1]
if len(args) > 2:
    nSamples = int(args[2])
else:
    nSamples = 1000

f = open(test, 'r')

# Execute the model file and load its variables into our globals
exec(f.read(), globals())
print('Testing: ', test, '--', testDescript)
print()
start = time.time()
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
    gnode = rv.RV(name, parents, observed, dType, None, None)
    gnodes.append(gnode)


# Constuct Gen with mod and sem params set as globals 'model' and 'varEquations'
# set into globals by executing the model file.
gen = gen_data.Gen(mod=model, sem=varEquations, globs=locals())

# Get the variable names in the same order as the samples will be generated
variables = gen.getVariables()
print('Vars = ', variables)

# Note that samples here is a generator function, so it, in effect, creates a
# stream that is only produced as it is consumed.
data = gen.getDataset(nSamples)

# For dat file, use the input file name with the .csv extension
tokens = test.split('.')
testFileRoot = str.join('.',tokens[:-1])
datFileName = testFileRoot + '.csv'

g = cGraph(gnodes, data)

#g.printGraph()

exos = g.findExogenous()
print()
print('Exogenous variables = ', exos)
print()
deps = g.computeDependencies(2)
g.printDependencies(deps)
print()

results = g.TestModel(order = 3, sensitivity=10)

conf, numTests, numTestsByType, numErrsByType, numWarnsByType, errors, warnings = results

print()
print('Confidence = ', conf)
print('Number of Tests = ', numTests)
print('Number of Tests by Type = ', numTestsByType)
print('Number of Errs by Type = ', numErrsByType)
print('Number of Warnings by Type = ', numWarnsByType)
print('Total Errors = ', len(errors))
print('Total Warnings = ', len(warnings))
print()
print('Error Details = ', errors)
print()
print('Warning Details = ', warnings)
print()

end = time.time()
duration = end - start
print('Test Time = ', round(duration))