import sys
from because.causality import rv, cGraph
from because.synth import read_data

from because.probability import independence

import time

args = sys.argv

if (len(args) > 1):
    test = args[1]
#test = 'C:/Users/91952/Because/because/probability/test/models/directTestDat.py'
f = open(test, 'r')

exec(f.read(), globals())

print('Testing: ', test, '--', testDescript)

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

# For dat file, use the input file name with the .csv extension

tokens = test.split('.')

testFileRoot = str.join('.', tokens[:-1])

datFileName = testFileRoot + '.csv'

d = read_data.Reader(datFileName)

data = d.read()

g = cGraph(gnodes, data)

g.printGraph()
'''
results = g.testAllDirections()
for result in results:
    print(result)
'''
edges = [
         ('N', 'N2'),
         ('N', 'N3'),
         ('M', 'M2'),
         ('M', 'M3'),
         ('M', 'M4'),
         ('IVB', 'IVA'),
         ('IVB', 'IVC'),
         ('EXP', 'EXP2')]
results = g.testAllDirections(edges)
#result = g.testAllDirections()
for result in results:
    print(result)

g.findExogenous()

end = time.time()

duration = end - start

print('Test Time = ', round(duration))