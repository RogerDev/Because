import sys

from because.causality import RV
from because.causality import cGraph
from because.synth import read_data
from because.probability import independence

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
    name, parents = var[:2]
    if len(var) >= 3:
        observed = var[2]
    if len(var) >= 4:
        dType = var[3]
    gnode = RV(name, parents, observed, dType, None, None)
    gnodes.append(gnode)

# For dat file, use the input file name with the .csv extension
tokens = test.split('.')
testFileRoot = str.join('.',tokens[:-1])
datFileName = testFileRoot + '.csv'

d = read_data.Reader(datFileName)
data = d.read()

g = cGraph(gnodes, data)

#g.printGraph()
pwr = 1

aBar = g.prob.distr('A').E()
print('E(A) ', aBar)
bBar = g.prob.distr('B').E()
print('E(B) ', bBar)
aStd = g.prob.distr('A').stDev()
print('std(A) = ', aStd)
bStd = g.prob.distr('B').stDev()
print('std(B) = ', bStd)

aHigh = aBar + .5 * aStd
aLow = aBar - .5 * aStd
print()
print('testRange = ', aHigh - aLow)
print('Test Bounds = [', aLow, ',', aHigh, ']')
print()
h1 = g.prob.distr('C', [('A', aHigh)]).E()
l1 = g.prob.distr('C', [('A', aLow)]).E()
print('E(C | A = aHigh)', h1)
print('E(C | A = aLow) = ', l1)
print('diff = ', h1 - l1)
print()
h2 = g.prob.distr('C', [('A', aHigh), 'B']).E()
l2 = g.prob.distr('C', [('A', aLow), 'B']).E()
print('E(C | A = aHigh,B)', h2)
print('E(C | A = aLow, B) = ', l2)
print('diff = ', h2 - l2)
print()

h3 = g.intervene('C', [('A', aHigh)], power=pwr).E()
l3 = g.intervene('C', [('A', aLow)], power=pwr).E()
print('E(C | do(A=aHigh)) = ', h3)
print('E(C | do(A=aLow)) = ', l3)
print('diff = ', h3 - l3)
print('implied ACE = ', (h3-l3)/ (aHigh - aLow))
print()
ace = g.ACE('A', 'C', power=pwr)
print('ACE(A, C) = ', ace)
print('ACE(C, A) = ', g.ACE('C', 'A'))
print()
cde = g.CDE('A', 'C', power=pwr)
print('Controlled Direct Effect(A, C) = ', cde)
print('Indirect Effect (A, C)', ace - cde)
#results = g.TestModel()
#conf = results[0]
#print('\nConfidence = ', round(conf * 100, 1), '%, testPerType = ', results[2], ', errsPerType = ', results[3])
#print('TestResults = ', results)
