from email.errors import MissingHeaderBodySeparatorDefect
import numpy as np
from math import sqrt, log, e, ceil
import copy

from because.probability.rkhs import rkhsRFF
from because.probability.rkhs import rkhsMV as Rmv

RKHS = rkhsRFF.RKHS
RKHSmv = Rmv.RKHS


epsilon = .001 # The maximum proability value considered to be zero
minDisc = 0 # The minimum number of points to use discretization.
              # If we have less than this number, use RKHS to estimate.
rmvCache = {} # Cache of rkhsMV by data and includevars

class PDF:
    """ Construct the PDF from a list of bins, where a bin is defined as:
        (binNumber, min, max, prob).
        binNumber is the zero based number of the bin (0 - numBins-1)
        min and max are the limits of data values within the bin min <= value < max
        prob is the probability of a point falling within the current bin.
        PDF normally uses only discretized data, but if "data" is provided,
        will use the full set of data.  This slows it down somewhat, and
        improves accuracy marginally.
    """
    def __init__(self, numSamples, binList=None, isDiscrete = False, data=None, mvData = None, conds = [], filters = [], rvName = None):
        self.N = numSamples
        self.bins = binList
        self.rvName = rvName
        self.filters = filters
        self.isDiscrete = isDiscrete
        self.binCount = 0
        self.style = 'd' # 'd' for discrete, 'r' for rkhs, 'mv' for multivariate
        #print('binList = ', binList, ', N = ', self.N)
        if binList:
            self.binCount = len(self.bins)
            totalP = 0.0
            for bin in binList:
                totalP += bin[3]
            #print('totalP = ', totalP, ', binCount = ', self.binCount, ', N = ', self.N)
            self.min = binList[0][1] # min of bin 0
            self.max = binList[self.binCount - 1][2] # max of last bin
        self.mvData = None
        self.filters = None
        if data is not None and not isDiscrete and self.N < minDisc:
            # Keep the data only if we want to use
            # RKHS.  Otherwise, we use the discetized
            # values.
            self.data = data
            self.R = RKHS(self.data)
            self.style = 'r'
        elif mvData is not None and not isDiscrete:
            self.style = 'mv'
            self.mvData = mvData
            self.filters = filters
            includeVars = [rvName]
            for filter in filters:
                includeVars.append(filter[0])
            self.smoothness = 1.0
            self.R = RKHSmv(self.mvData, includeVars, s=self.smoothness)
            self.R2 = RKHSmv(self.mvData, includeVars[1:], s=self.smoothness)
            # Min and max can't be outside the bounds of the unconditioned data.
            varData = self.mvData[rvName]
            self.min = min(varData)
            self.max = max(varData)
        else:
           # Drop the data and use discretized.
            self.R = None
            self.data = None
        #print('pdf:  style = ', self.style)
        self.meanCache = None
        self.varCache = None
        self.skewCache = None
        self.kurtCache = None

    def binValue(self, i):
        bin = self.bins[i]
        min, max = bin[1:3]
        if self.isDiscrete:
            value = min
        else:
            value = (min + max) / 2
        return value

    def minVal(self):
        return self.min
    
    def maxVal(self):
        return self.max

    def getBinForVal(self, value):
        if value < self.min:
            return 0
        elif value >= self.max:
            return len(self.bins)-1
        for i in range(self.binCount):
            bin = self.bins[i]
            indx, start, end, prob = bin
            if value >= start and value < end:
                return i

    def P(self, valueSpec):
        """ Return the probability of a given value or range of values.
            valueSpec can be:
            number -- The probability of attaining the given value
                        Note: For real valued continuous variables, the probability
                        of attaining a given value is essentially zero.  In this case,
                        the probability of attaining a value within the discretized bin
                        associated with value is returned.  This is useful for comparison
                        purposes, but the returned probability is dependent on the discretization
                        density.
            (low, high) -- The probability of attaining a value in the range (low <= value < high)
                        high = None means infinity
                        low = None means negative infinity
        """
        value = None
        if type(valueSpec) == type((0,)) and len(valueSpec) > 1:
            # it's a range tuple
            assert len(valueSpec) == 2, 'pdf.P: valueSpec must be a single number or 2-tuple = ' + str(valueSpec)
            low, high = valueSpec
            if low is None:
                low = self.min - 1
            if low > self.max:
                return 0.0
            if high is None:
                high = self.max + 1
            if high <= self.min:
                return 0.0
        elif type(valueSpec) == type((0,)):
            value = valueSpec[0]
        else:
            value = valueSpec

        if self.style == 'r':
            print('pdf: using rkhs')
            if value is not None:
                # Exact query
                p = self.R.F(value)
            else:
                p = self.R.F(high, 'c') - self.R.F(low, 'c')
            return p
        elif self.style == 'mv':
            if value is not None:
                # Exact query
                print('pdf--mv: single value')
                includeVals = [value]
                for f in self.filters:
                    if type(f) == type((0,) and len(f) > 2):
                        var, low, high = f
                        value = (low + high) / 2.0
                    elif type(f) == type((0,)):
                        var, value = f
                    else:
                        print('bad filter')
                    includeVals.append(value)
                p = self.R.condP(includeVals)
            else:
                print('pdf--mv: range')
                
                includeVals1 = [high]
                includeVals2 = [low]
                for f in self.filters:
                    if type(f) == type((0,) and len(f) > 2):
                        var, low, high = f
                        value = (low + high) / 2.0
                    elif type(f) == type((0,)):
                        var, value = f
                    else:
                        print('bad filter')
                    includeVals1.append(value)
                    includeVals2.append(value)
                p = self.R.condCDF(includeVals1) - self.R.condCDF(includeVals2)

            return p

        elif False:
            # We have the actual data. Do an exact calculation.
            matches = 0
            if value is None:
                # Range match
                for i in range(self.N):
                    val = self.data[i]
                    if val >= low and val < high:
                        matches += 1
            else:
                # Exact match
                for i in range(self.N):
                    val = self.data[i]
                    if val == value:
                        matches += 1
                #print('matches = ', matches)
            prob = matches / self.N
            return prob
        else:
            # Calculate based on discretized bins
            if value is not None:
                if value < self.min or value >= self.max:
                    return 0.0  # Outside the range.  Zero probability.
                # Value match
                outProb = 0.0
                for i in range(self.binCount):
                    bin = self.bins[i]
                    indx, start, end, prob = bin
                    if value >= start and value < end:
                        outProb = prob / (end-start)
                        break
                return outProb
            else:
                # Range match
                firstBin = self.getBinForVal(low)
                lastBin = self.getBinForVal(high)
                cum = 0.0
                return self.Prangex(low, high)


    def Prangex(self, minVal, maxVal):
        """ Return the probability of x between 2 values
            i.e. P(minVal <= X < maxVal)
        """
        if minVal < self.min:
            minVal = self.min
        if maxVal > self.max:
            maxVal = self.max
        firstBin = self.getBinForVal(minVal)
        lastBin = self.getBinForVal(maxVal)
        cum = 0.0
        for i in range(firstBin, lastBin + 1):
            bin = self.bins[i]                
            indx, bmin, bmax, prob = bin
            if i == firstBin and i == lastBin:
                adjProb = (maxVal - minVal) / (bmax - bmin) * prob
            elif i == firstBin:
                adjProb = (bmax - minVal) / (bmax - bmin) * prob
            elif i == lastBin:
                adjProb = (maxVal - bmin) / (bmax - bmin) * prob
            else:
                adjProb = prob
            cum += adjProb
        return cum

    def E(self):
        """Return the expected value (e.g. mean) of the disribution."""
        #print('pdf.E')
        if self.meanCache is None:
            if self.style == 'r':
                #print('pdf.E: Using rkhs')
                result = self.R.F(None, 'e')
                #print('pdf.E: returning rkhs result = ', result)
            elif self.style == 'mv':
                #print('pdf: using mv')
                includeVals = []
                for f in self.filters:
                    if type(f) == type((0,) and len(f) > 2):
                        var, low, high = f
                        value = (low + high) / 2.0
                    elif type(f) == type((0,)):
                        var, value = f
                    else:
                        print('bad filter')
                    includeVals.append(value)
                result  = self.R2.condE(self.rvName, includeVals)
            else:
                #print('pdf.E: Bypassing rkhs')
                cum = 0
                for i in range(self.binCount):
                    bin = self.bins[i]
                    id, min, max, prob = bin
                    value = self.binValue(i)
                    cum += prob * value
                result = cum
            self.meanCache = result
            return result
        else:
            return self.meanCache

    mean = E
    
    def mode(self):
        #assert self.isDiscrete, 'Mode is only available for discrete variables'
        maxProb = 0.0
        maxVal = None
        for i in range(self.binCount):
            bin = self.bins[i]
            prob = bin[3]
            if prob > maxProb:
                maxProb = prob
                maxVal = self.binValue(i)
        return maxVal

    def modality(self):
        """
        Return the number of modes of the disribution (e.g. uni-modal, bi-modal, n-modal)
        """
        minProb = .4 / self.binCount
        #print('minProb = ', minProb)
        thresh = 1.4
        modes = 0
        prevPeak = 0
        prevTrough = 1
        direction = 0
        for i in range(self.binCount):
            bin = self.bins[i]
            prob = bin[3]
            #print('prob, direction, prevPeak, prevTrough = ', prob, direction, prevPeak, prevTrough)
            if prob > minProb:
                if  direction >= 0:
                    if prob > prevPeak:
                        prevPeak = prob
                        direction = 1
                    elif prob < prevPeak / thresh:
                        modes += 1
                        direction = -1
                        prevTrough = prob
                elif direction < 0:
                    if prob < prevTrough:
                        prevTrough = prob
                    if prob > prevTrough * thresh:
                        direction = 1
                        prevPeak = prob
        return modes

    def truncation(self):
        """
        Determine whether a distributed is truncated (bounded) at either
        or both (upper and lower) ends, and what those bounds are.
        Returns (lowerBound, upperBound), where the bounds are specified
        as a location, or None for unbounded.
        """
        bin0 = self.bins[0]
        binN = self.bins[-1]
        prob0 = bin0[3]
        probN = binN[3]
        if prob0 > .1 / self.binCount:
            lower = int(self.minVal() * 100) / 100.0
        else:
            lower = None
        print('bnN = ', binN, ', maxVal = ', self.maxVal())
        if probN > .1 / self.binCount:
            upper = ceil(self.maxVal() * 100) / 100.0
        else:
            upper = None
        return (lower, upper)

    def percentile(self, ptile):
        #assert False, 'BinList = ' + str(self.bins)
        ptile2 = ptile / 100.0
        cum = 0.0
        if ptile == 0:
            return self.minVal()
        if ptile == 100:
            return self.maxVal()
        val = None
        for i in range(self.binCount):
            bin = self.bins[i]
            prob = bin[3]
            if cum + prob >= ptile2:

                if self.isDiscrete:                    
                    # Use Nearest Rank method
                    # Return the value of the first bin cumulating at least that percentage of points.
                    val = bin[1]
                else:
                    # Linearly interpolate 
                    binMin = bin[1]
                    binMax = bin[2]
                    binRange = binMax - binMin
                    pctMin = cum
                    pctMax = cum + prob
                    pctRange = pctMax - pctMin
                    pctPos = (ptile2 - pctMin) / pctRange
                    val = binMin + pctPos * binRange
                break
            else:
                cum += prob
        return val
   
    def median(self):
        return self.percentile(50.0)
    
    def var(self):
        if self.varCache is None:
            mean = self.E()
            cum = 0.0
            for i in range(self.binCount):
                bin = self.bins[i]
                prob = bin[3]
                value = self.binValue(i)
                cum += prob * (value - mean)**2
            var = cum
            # Generate the sample variance by multiplying by N / (N-1)
            if self.N > 1:
                sv = var * self.N / (self.N - 1)
            else:
                sv = var
                print('PDF.var: Warning -- Sample too small to compute sample variance = ', self.N)
            self.varCache = sv
            return sv
        else:
            return self.varCache

    def stDev(self):
        var = self.var()
        std = sqrt(var)
        return std

    def skew(self):
        if self.skewCache is None:
            mean = self.E()
            std = self.stDev()
            if std == 0:
                return 0.0
            cum = 0.0
            for i in range(self.binCount):
                bin = self.bins[i]
                id, min, max, prob = bin
                value = self.binValue(i)
                cum += prob * ((value-mean) / std)**3
            # Generate sample skew by applying correction based on N
            if self.N > 2:
                ssk = cum * self.N**2 / ((self.N - 1) * (self.N - 2))
            else:
                ssk = cum
                print('PDF.skew: Warning -- Sample too small to compute sample skew = ', self.N)
            self.skewCache = ssk
            return ssk
        else:
            return self.skewCache

    def kurtosis(self):
        """ Return the excess kurtosis of the distribution"""
        if self.kurtCache is None:
            mean = self.E()
            std = self.stDev()
            if std == 0:
                return 0.0
            cum = 0.0
            for i in range(self.binCount):
                bin = self.bins[i]
                id, min, max, prob = bin
                value = self.binValue(i)
                cum += prob * ((value-mean) / std)**4
            ekurt = cum - 3.0 # Excess Kurtosis
            self.kurtCache = ekurt
            return ekurt
        else:
            return self.kurtCache

    def stats(self):
        m1 = self.mean()
        m2 = self.stDev()
        m3 = self.skew()
        m4 = self.kurtosis()
        return(self.N, m1, m2, m3, m4)

    def ToHistogram(self):
        """Convert the pdf to a numpy array of probabilities [P(bin1), ..., P(binN)]"""
        return np.array([bin[3] for bin in self.bins])

    def ToHistTuple(self):
        """Convert the pdf to a list of tuples of binVal, and P(bin) --  [(binVal1, P(bin1)), ..., (binValN, P(binN))]"""
        outTups = []
        for i in range(len(self.bins)):
            bin = self.bins[i]
            binProb = bin[3]
            if self.isDiscrete:
                outTups.append((bin[1], bin[1], binProb))
            else:    
                outTups.append((bin[1], bin[2], binProb))
        return outTups

    def SetHistogram(self, newHist):
        assert len(newHist) == len(self.bins), "PDF.SetHistogram: Cannot set histogram with different lenght than current distribution.  (new, original) = " + str((len(newHist), len(self.bins)))
        outBins = []
        for i in range(len(self.bins)):
            bin = self.bins[i]
            prob = newHist[i]
            newBin = bin[:-1] + (prob,)
            outBins.append(newBin)
        self.bins = outBins

    def getBin(self, binNum):
        return self.bins[binNum]

    def __add__(self, other):
        sHist = self.ToHistogram()
        oHist = other.ToHistogram()
        outHist = sHist + oHist
        out = copy.copy(self)
        out.SetHistogram(outHist)
        return out

    def __sub__(self, other):
        sHist = self.ToHistogram()
        oHist = other.ToHistogram()
        outHist = sHist - oHist
        out = copy.copy(self)
        out.SetHistogram(outHist)
        return out

    def __mult__(self, other):
        sHist = self.ToHistogram()
        oHist = other.ToHistogram()
        outHist = sHist * oHist
        out = copy.copy(self)
        out.SetHistogram(outHist)
        return out

    def compare2(self, other):
        # Bin Comparison Method
        assert len(self.bins) == len(other.bins), "PDF.compare():  Bin sizes must match for each distribution " + str((len(self.bins), len(other.bins)))
        accum = 0.0
        errs = []
        for i in range(len(self.bins)):
            bin1 = self.bins[i]
            bin2 = other.bins[i]
            prob1 = bin1[3]
            prob2 = bin2[3]
            diff = abs(prob1 - prob2) / 2
            #print('prob1, prob2, diff = ', i, prob1, prob2, diff)
            accum += diff
            errs.append(diff)
        return accum

    def compare4(self, other):
        # Range Comparison Method
        testRanges = 10
        accum = 0.0
        minVal1 = self.minVal()
        minVal2 = other.minVal()
        maxVal1 = self.maxVal()
        maxVal2 = other.maxVal()
        minVal = min([minVal1, minVal2])
        maxVal = max([maxVal1, maxVal2])
        #print('minVal, maxVal = ', minVal, maxVal)
        ranges = list(np.arange(minVal, maxVal + .00001, (maxVal - minVal) / float(testRanges)))
        #print('ranges =', len(ranges), ranges)
        errs = []
        for i in range(testRanges):
            r1 = ranges[i]
            r2 = ranges[i+1]
            p1 = self.P((r1, r2))
            p2 = other.P((r1,r2))
            diff = abs(p1 - p2) * (p1 + p2) / 2 
            #print('i, N1, N2, r1, r1, p1, p2, diff = ', i, self.N, other.N, r1, r2, p1, p2, diff)
            accum += diff
            errs.append(diff)
        return accum

    def compare3(self, other):
        # Moment comparison method
        assert len(self.bins) == len(other.bins), "PDF.compare():  Bin sizes must match for each distribution " + str((len(self.bins), len(other.bins)))
        mean1 = self.mean()
        mean2 = other.mean()
        std1 = self.stDev()
        std2 = other.stDev()
        dep1 = abs((mean1 - mean2))
        dep2 = abs((std1 - std2) / (std1 + std2))
        sk1 = self.skew()
        sk2 = other.skew()
        ku1 = self.kurtosis()
        ku2 = other.kurtosis()
        dep3 = abs(sk1 - sk2)
        dep4 = abs(ku1 - ku2)
        dep3 = 0
        dep4 = 0
        dep = max([dep1, dep2, dep3, dep4])
        return dep

    def compare5(self, other):
        # Z-statistic
        mean1 = self.mean()
        mean2 = other.mean()
        std1 = self.stDev()
        std2 = other.stDev()
        sqrtN1 = sqrt(self.N)
        sqrtN2 = sqrt(other.N)
        Z = abs(mean1-mean2) / sqrt((std1/sqrtN1)**2 + (std2/sqrtN2)**2)
        dep = Z
        return dep

    # Compare distributions based on the Komogorov-Smirnov Test Statistic
    ksAlpha = .5 # Confidence level for KS Test.  At this level, the null
                # hypothesis of independece is rejected.
    ksThreshold = sqrt(-log(ksAlpha/2) * .5) # Threshold for the ksTest

    def compare(self, other, raw=False):
        # Kolmogorov-Smirnov Statistic
        if self.isDiscrete:
            testRanges = self.binCount
            #print('pdf.compare(discrete): self.N, other.N = ', self.N, other.N)
        else:
            minN = min([self.N, other.N])
            testRanges = min([max([int(minN/10), 5]), 20])
            #print('testRanges = ', testRanges, self.N, other.N)
            minVal1 = self.minVal()
            minVal2 = other.minVal()
            maxVal1 = self.maxVal()
            maxVal2 = other.maxVal()
            minVal = min([minVal1, minVal2])
            maxVal = max([maxVal1, maxVal2])
            #print('minVal, maxVal = ', minVal, maxVal)
            ranges = list(np.arange(minVal, maxVal + .00001, (maxVal - minVal) / float(testRanges)))
        #print('ranges =', len(ranges), ranges)
        N1 = self.N
        N2 = other.N
        minN = min([N1, N2])
        cdf1 = 0.0
        cdf2 = 0.0
        diffs = []
        for i in range(testRanges):
            if self.isDiscrete:
                val = self.binValue(i)
                p1 = self.P(val)
                p2 = other.P(val)
                cdf1 += p1
                cdf2 += p2
            else:
                r1 = ranges[i]
                r2 = ranges[i+1]
                p1 = self.P((r1, r2))
                p2 = other.P((r1,r2))
                cdf1 += p1
                cdf2 += p2
            diff = abs(cdf1 - cdf2)
            diffs.append(diff)
        ks = max(diffs)
        if raw:
            #mean1 = self.mean()
            #mean2 = other.mean()
            #meanDiff = abs(mean1-mean2)
            #std1 = self.stDev()
            #std2 = other.stDev()
            #stdDiff = abs(std1-std2)
            #return ks + meanDiff + stdDiff
            return ks
        # Now, convert the raw KS statistic (i.e. D(m,n)) to an alpha
        # value, where alpha := P(self != other).  That is, the probability
        # that self and other were not drawn from the same distribution.
        # This is an inversion of the usual use of KS where comparing D to
        # an alpha based threshold.
        #alpha = 2 * e**(-2 *(ks/sqrt((N1+N2)/ (N1*N2)))**2)
        alpha = 2 * e**(-2 *(ks/sqrt((2*minN)/ (minN*minN)))**2)
        #print('N1, N2, D, alpha = ', N1, N2, ks, alpha)
        return max([0, 1-alpha])

    def compare_ks_old(self, other):
        # Kolmogorov-Smirnov Statistic
        testRanges = 10
        minVal1 = self.minVal()
        minVal2 = other.minVal()
        maxVal1 = self.maxVal()
        maxVal2 = other.maxVal()
        minVal = min([minVal1, minVal2])
        maxVal = max([maxVal1, maxVal2])
        #print('minVal, maxVal = ', minVal, maxVal)
        ranges = list(np.arange(minVal, maxVal + .00001, (maxVal - minVal) / float(testRanges)))
        #print('ranges =', len(ranges), ranges)
        N1 = self.N
        N2 = other.N
        cdf1 = 0.0
        cdf2 = 0.0
        diffs = []
        for i in range(testRanges):
            r1 = ranges[i]
            r2 = ranges[i+1]
            p1 = self.P((r1, r2))
            p2 = other.P((r1,r2))
            cdf1 += p1
            cdf2 += p2
            #if p1 > 0 and p2 > 0:
            diff = abs(cdf1 - cdf2)
            diffs.append(diff)
        ks = max(diffs)
        # Normalize the ks statistic based on the number of data points
        # and the desired threshold (i.e. ksThreshold).
        # The .5 multiplier normalizes to .5 rather than to 1.0 as the final
        # threshold.
        normKS = ks / (self.ksThreshold * sqrt((N1 + N2) / (N1 * N2))) * .5
        return normKS


    def isNull(self):
        for val in self.ToHistogram():
            if val > epsilon:
                return False
        return True

    def __eq__(self, other):
        return (self - other).isNull()

