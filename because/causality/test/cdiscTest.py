from because.probability import ProbSpace
from because.synth import gen_data
from because.synth import read_data
from because.causality import cdisc
import time
from sys import argv

datSize = 100000

start = time.time()
if len(argv) >= 2:
    testFile = argv[1]
else:
    testFile = 'causality/test/models/scanData1.py'

print()
print('Using test file = ', testFile)
gen = gen_data.Gen(testFile)
data = gen.getDataset(datSize)

ps = ProbSpace(data)
power = 10
maxLevel = 2
results = cdisc.discover(ps, maxLevel=maxLevel, power=power)


end = time.time()
elapsed = end - start
print()
for rv in results:
    print(rv.name, rv.parents)
print()
print('elapsed = ', round(elapsed, 0))
