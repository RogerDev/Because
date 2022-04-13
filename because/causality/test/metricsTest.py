import sys

from because.causality import RV
from because.causality import cGraph
from because.synth import read_data, gen_data
from because.probability import independence

DEBUG = True

quiet = False
pwr = 1
datSize = 10000

args = sys.argv
if (len(args) > 1):
    datSize = int(args[1])

test = 'causality/test/models/metricsTestDat.py'

f = open(test, 'r')
exec(f.read(), globals())
if not quiet:
    print('Testing: ', test, '--', testDescript)

# For dat file, use the input file name with the .csv extension
tokens = test.split('.')
testFileRoot = str.join('.',tokens[:-1])
datFileName = testFileRoot + '.csv'

gen = gen_data.Gen(test)
sdg = gen.generate(datSize)
d = read_data.Reader(datFileName, quiet=quiet)
data = d.read()


# 'model' is set when the text file is exec'ed
varNames = []
gnodes = []
for var in model:
    observed = True
    dType = 'Numeric'
    name, parents = var[:2]
    varNames.append(name)
    if len(var) >= 3:
        observed = var[2]
    if len(var) >= 4:
        dType = var[3]
    gnode = RV(name, parents, observed, dType, None, None)
    gnodes.append(gnode)

# For dat file, use the input file name with the .csv extension


g = cGraph(gnodes, data)

varNames.sort()

#dist = g.intervene('C', [('A',1), ('B', 1)])
#print('Intervention: Mean = ', dist.E())

testNum = 1
for var1 in varNames:
    for var2 in varNames:
        #if testNum > 2:
        #    break
        if var1 == var2:
            continue
        tstr = var1 + ' -> ' + var2
        if DEBUG:
            print('\nTesting ' + tstr)
        ace = g.ACE(var1, var2, power=pwr)
        cde = g.CDE(var1, var2, power=pwr)
        cie = ace - cde
        print(tstr, ': ACE =', ace, ', CDE =', cde, ', CIE =', cie)
        testNum += 1


