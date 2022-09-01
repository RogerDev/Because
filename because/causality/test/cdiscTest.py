from because.probability import ProbSpace
from because.synth import gen_data
from because.synth import read_data
from because.causality import cdisc
import time
from sys import argv

datSize = 100000

start = time.time()
power = 5
sensitivity = 5
tSpec = None
if len(argv) >= 2:
    dataPath = argv[1]
else:
    dataPath = 'causality/test/models/scanData1.py'
if len(argv) >= 3:
    targets = argv[2].strip()
    tokens = targets.split(',')
    tSpec = []
    for token in tokens:
        varName = token.strip()
        if varName:
            tSpec.append(varName)

if len(argv) >= 4:
    power = int(argv[3])
if len(argv) >= 5:
    sensitivity = int(argv[4])

print()
# Got a .csv or .py file
tokens = dataPath.split('.')
ds = None # The dataset in dictionary form
assert len(tokens) == 2 and (tokens[1] == 'py' or tokens[1] == 'csv'), 'cdiscTest: dataPath must have a .py or .csv extension.  Got: ' + dataPath
if tokens[1] == 'py':
    # py SEM file
    gen = gen_data.Gen(dataPath)
    ds = gen.getDataset(datSize)
else:
    # csv
    r = read_data.Reader(dataPath)
    ds = r.read()
ps = ProbSpace(ds, power=power)
maxLevel = 2
results = cdisc.discover(ps, varNames=tSpec, maxLevel=maxLevel, power=power, sensitivity=sensitivity)


end = time.time()
elapsed = end - start
print()
for rv in results:
    print(rv.name, rv.parentNames)
print()
print('elapsed = ', round(elapsed, 0))
