import numpy as np
from math import ceil

class Grid:
    def __init__(self, ps, vars, lim=1, numPts=20):
        dims = len(vars)
        assert dims > 0 and dims <= 3, 'makeGrid:  Only dimensionality 1 through 3 is supported.  Got: ' +  str(dims)
        distrs = [ps.distr(c) for c in vars]
        minvs = [distr.percentile(lim) for distr in distrs]
        maxvs = [distr.percentile(100-lim) for distr in distrs]
        # Each vSpace entry is (nominalVal, val) or (nominalVal, minVal, maxVal)
        vSpaces = []
        incrs = []
        nTests = 1
        for i in range(dims):
            nSamples = numPts
            vSpace = []
            var = vars[i]
            if ps.isDiscrete(var):
                allVals = ps.getMidpoints(var)
                if ps.isCategorical(var):
                    for val in allVals:
                        # Use all values.  Nominal = val
                        vSpace.append((val, val))
                else:
                    # Discrete numeric.  Sample a range of values
                    allVals0 = ps.getMidpoints(var)
                    allVals = []
                    minv = minvs[i]
                    maxv = maxvs[i]
                    # Bound the values between minv and maxv
                    for val in allVals0:
                        if val >= minv and val <= maxv + .01:
                            allVals.append(val)
                    nVals = len(allVals)
                    #print('nVals = ', nVals)
                    if nVals <= nSamples:
                        testVals = allVals
                        sampleIndxs = list(range(0, nVals))
                        # We may return fewer samples than requested
                        nSamples = len(sampleIndxs)
                    else:
                        # Let's try and reduce the number of values.
                        reduction = int(nVals / nSamples)
                        # Take every Kth value, where K = reduction
                        sampleIndxs = list(range(0, nVals, reduction))
                        # We may return more samples than requested since we can only reduce by whole numbers 
                        nSamples = len(sampleIndxs)
                        testVals = [allVals[indx] for indx in sampleIndxs]
                        #print('reduction = ', reduction, ', testVals = ', len(testVals), testVals)
                    for j in range(len(testVals)-1):
                        testVal = testVals[j]
                        nextVal = testVals[j+1]
                        nominal = (testVal + nextVal) * .5
                        vSpace.append((nominal, testVal, nextVal))
            else:
                # Continuous.  Use even ranges.
                testVals = list(np.linspace(minvs[i], maxvs[i], numPts + 1))[:-1]
                for j in range(len(testVals)):
                    testVal = testVals[j]
                    if j == len(testVals) - 1:
                        midPt = (testVal + maxvs[i]) * .5
                        vSpace.append((midPt, testVal, None))
                    else:
                        next = testVals[j+1]
                        midPt = (testVal + next) * .5
                        vSpace.append((midPt, testVal, next))
            nTests *= len(vSpace)
            vSpaces.append(vSpace)
        self.dims = dims
        self.vSpaces = vSpaces
        self.nTests = nTests

    def getTestCount(self, varNum=None):
        if varNum is not None:
            return len(self.vSpaces[varNum])
        return self.nTests

    def makeGrid(self):
        vSpaces = self.vSpaces
        dims = self.dims
        for val1 in vSpaces[0]:
            if dims > 1:
                for val2 in vSpaces[1]:
                    if dims > 2:
                        for val3 in vSpaces[2]:
                            yield ((val1, val2, val3))
                    else:
                        yield ((val1, val2))
            else:
                yield((val1,))
    
    
                
        


