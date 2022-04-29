
from because.synth import gen_data
from because.synth import read_data
from because.causality import cscan
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
i = 0

scanner = cscan.Scanner(data)

results = scanner.scan()
end = time.time()
elapsed = end - start
print()
print(results)
print()
print('elapsed = ', round(elapsed, 0))
