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
        nSamples = numPts - 1
        for i in range(dims):
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
                    allVals = ps.getMidpoints(var)
                    nVals = len(allVals)
                    if nVals <= nSamples:
                        # Eliminate the last value. We'll add in later.
                        testVals = allVals[:-1]
                        sampleIndxs = list(range(0, nVals))
                    else:
                        # Let's try and reduce the number of values.
                        reduction = int(nVals / numPts)
                        # Take every Kth value, where K = reduction
                        sampleIndxs = list(range(0, nVals, reduction))
                        # Since the above sometimes loses the last value (when nVals is even),
                        # we'll take the center nSamples out of the remaining values, but 
                        # when the remaining values - nSamples is odd, we'll favor the later values.
                        start = ceil((len(sampleIndxs) - nSamples) / 2)
                        # Extract the nSamples center values.
                        sampleIndxs = sampleIndxs[start : start + nSamples]
                        testVals = [allVals[indx] for indx in sampleIndxs]
                    for j in range(len(testVals)):
                        testVal = testVals[j]
                        if j == 0:
                            # First interval is [minVal, testVal]
                            next = allVals[sampleIndxs[j] + 1]
                            nominal = (allVals[0] + testVal) * .5
                            vSpace.append((nominal, None, next))
                        else:
                            # Interval is (prev, testVal]
                            prev = allVals[sampleIndxs[j - 1] + 1]
                            if sampleIndxs[j] < len(allVals) - 1:
                                next = allVals[sampleIndxs[j] + 1]
                            else:
                                next = allVals[-1] + 1
                            #prev = testVals[j - 1]
                            nominal = (prev + testVal) * .5
                            vSpace.append((nominal, prev, next))
                    # Always add in the last test val up to the max val
                    # Interval is (prev, maxVal]
                    maxVal = allVals[-1]
                    if len(sampleIndxs) == len(allVals):
                        lastTest = allVals[-1]
                    else:
                        lastTest = allVals[sampleIndxs[-1] + 1]
                    nominal = (maxVal + lastTest) * .5
                    vSpace.append((nominal, lastTest, None))
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
    
    
                
        


