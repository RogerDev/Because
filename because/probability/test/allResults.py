import sys
if '.' not in sys.path:
    sys.path.append('.')

from because.probability.test import cprobEval

dims = [2,3,4,5,6]
#dims = [5,6]
datSizes = [10,100,1000,10000, 100000, 1000000]
#datSizes = [1000000]

triesDict = {10:50, 100:20, 1000:10, 10000:3, 100000:1, 1000000:1}
fname = 'allResults.csv'
f = open(fname, 'w')
f.write(",,,,R-Squared,,,Runtime,\n")
f.write("N, D, Runs, D-Prob, J-Prob, U-Prob, D-Prob, J-Prob, U-Prob\n")
f.flush()
for datSize in datSizes:
    for dim in dims:
        tries = triesDict[datSize]
        result = cprobEval.run(dim, datSize, tries, quiet=True)
        f.write(repr(result)[1:-1]+ '\n')
        f.flush()
f.close()