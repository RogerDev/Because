import sys
from because.causality import rv, cGraph
from because.synth import read_data
from because.synth import gen_data
import numpy as np

from because.probability import independence

import time

args = sys.argv

if (len(args) > 1):
    test = args[1]
test = 'models/directTestDat.py'
dataPoints = 10000
runs = 10
sensitivity = 1

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
         ('M', 'M6'),
         ('M6', 'M'),
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
n = len(edges)
res = np.zeros((n, runs))

for run in range(runs):
    gen = gen_data.Gen(test)
    dat = gen.getDataset(dataPoints)

    g = cGraph(gnodes, dat)

    results = g.testAllDirections(edges, power=2, sensitivity=sensitivity)

    for i in range(n):
        _, _, _, res[i, run] = results[i]

#g.findExogenous()
res = res.mean(axis=1)
total_margin = 0
correct = 0
for i in range(n):
    if i % 2 == 1:
        print(edges[i-1])
        print(f'forward: {res[i-1]}')
        print(f'backward: {res[i]}')
        print(f'margin: {res[i-1] - res[i]}')
        total_margin += res[i-1] - res[i]
        if res[i-1] - res[i] > 0.0002:
            correct += 1

print("--------------------------")
print("--------------------------")
print(f'Total Margin: {total_margin} | Correct: {correct}')

end = time.time()

duration = end - start

print('Test Time = ', round(duration))