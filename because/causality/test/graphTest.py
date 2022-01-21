"""
    Present a plot of the distributions for the given .py test file
    python3 Probabiity/probPlot.py <testfilepath>.py
    Data should previously have been generated using:
    python3 synth/synthDataGen.py <testfilepath>.py <numRecs>
"""
import sys
if '.' not in sys.path:
    sys.path.append('.')
    
from because.causality import rv
from because.causality import cgraph
from because.synth import read_data
from because.probability import independence
import time

args = sys.argv
if (len(args) > 1):
    test = args[1]

f = open(test, 'r')
exec(f.read(), globals())

print('Testing: ', test, '--', testDescript)

gnodes = []

# 'model' is set when the text file is exec'ed
for var in model:
    observed = True
    dType = 'Numeric'
    if type(var) == type((1,)):
        name, parents = var[:2]
        if len(var) >= 3:
            observed = var[2]
        if len(var) >= 4:
            dType = var[3]
    else:
        name = var
        parents = []
        observed = True
        dType = 'Numeric'
    gnode = rv.RV(name, parents, observed, dType, None, None)
    gnodes.append(gnode)

# For dat file, use the input file name with the .csv extension
tokens = test.split('.')
testFileRoot = str.join('.',tokens[:-1])
datFileName = testFileRoot + '.csv'

d = read_data.Reader(datFileName)
data = d.read()

g = cgraph.cGraph(gnodes, data)

g.printGraph()

g.prob.Plot()
