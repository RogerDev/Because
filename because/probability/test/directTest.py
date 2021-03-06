import sys
from because.causality import rv, cGraph
from because.synth import read_data

from because.probability import independence

import time

args = sys.argv

if (len(args) > 1):
    test = args[1]
test = 'C:/Users/91952/Because/because/probability/test/models/directTestDat.py'
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
         ('N2', 'N'),
         ('N', 'N3'),
         ('N3', 'N'),
         ('M', 'M2'),
         ('M2', 'M'),
         ('M', 'M3'),
         ('M3', 'M'),
         ('M', 'M4'),
         ('M4', 'M'),
         ('M', 'M5'),
         ('M5', 'M'),
         ('IVB', 'IVA'),
         ('IVA', 'IVB'),
         ('IVB', 'IVC'),
         ('IVC', 'IVB'),
         ('EXP', 'EXP2'),
         ('EXP2', 'EXP'),
         ('EXP', 'EXP3'),
         ('EXP3', 'EXP'),
         ('EXP', 'EXP4'),
         ('EXP4', 'EXP'),
         ('EXP2', 'EXP4'),
         ('EXP4', 'EXP2'),
         ('EXP', 'EXP5'),
         ('EXP5', 'EXP'),
         ('EXP2', 'EXP5'),
         ('EXP5', 'EXP2')
]

results = g.testAllDirections(edges, power=2)
#result = g.testAllDirections()
for result in results:
    print(result)

#g.findExogenous()

end = time.time()

duration = end - start

print('Test Time = ', round(duration))