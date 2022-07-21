import numpy as np

class Grid:
    def __init__(self, ps, vars, lim=1, numPts=20):
        dims = len(vars)
        assert dims > 0 and dims <= 3, 'makeGrid:  Only dimensionality 1 through 3 is supported.  Got: ' +  str(dims)
        distrs = [ps.distr(c) for c in vars]
        minvs = [distr.percentile(lim) for distr in distrs]
        maxvs = [distr.percentile(100-lim) for distr in distrs]
        incrs = [(maxvs[i] - minvs[i]) / (numPts-1) for i in range(dims)]
        vSpaces = []
        incrs = []
        nTests = 1
        for i in range(dims):
            var = vars[i]
            if ps.isDiscrete(var):
                #print('var', var, 'is Discrete')
                vSpace0 = ps.getMidpoints(var)
                categorical = ps.isCategorical(var)
                if not categorical:
                    dist = distrs[i]
                    minBin = dist.getBinForVal(minvs[i])
                    maxBin = dist.getBinForVal(maxvs[i])
                    vSpace0 = vSpace0[minBin:maxBin+1]
                    reductAmount = max([1, int(len(vSpace0) /numPts)])
                    #print('reductAmount = ', reductAmount)
                    vSpace = []
                    for i in range(len(vSpace0)):
                        if i % reductAmount == 0:
                            vSpace.append(vSpace0[i])
                    incrs.append(reductAmount + 1)
                else:
                    vSpace = vSpace0
                    incrs.append(1)
            else:
                #print('var', var, 'not Discrete')
                #bins = distrs[i].bins
                #print('bins = ', len(bins), bins[:2])
                vSpace = list(np.linspace(minvs[i], maxvs[i], numPts))
                incrs.append(vSpace[1] - vSpace[0])
            nTests *= len(vSpace)
            vSpaces.append(vSpace)
        self.dims = dims
        self.vSpaces = vSpaces
        self.incrs = incrs
        self.nTests = nTests

    def getTestCount(self, varNum=None):
        if varNum is not None:
            return len(self.vSpaces[varNum])
        return self.nTests

    def getIncrs(self):
        return self.incrs

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
    
    
                
        


