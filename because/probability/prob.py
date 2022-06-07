# Probability Space Module provides all Probability Layer capabilities.

import numpy as np
import math
from math import log, sqrt
try:
    from because.probability import probCharts
except:
    pass
from because.probability.pdf import PDF
from because.probability import uprob

DEBUG = False

class ProbSpace:
    def __init__(self, ds, density = 1.0, power=1, discSpecs = None, cMethod = 'd!'):
        """ Probability Space (i.e. Joint Probability Distribution) based on a a multivariate dataset
            of random variables provided.  'JPS' (Joint Probability Space) is an alias for ProbSpace.
            The Joint Probability Space is a multi-dimensional probability distribution that embeds all
            knowledge about the statistical relationships among the variables, and supports a powerful
            range of queries to expose that information.
            It can handle discrete as well as continuous variables.  Continuous probabilities
            are managed by dense discretization (binning) continuous variables into small ranges.
            By default, the number of discretized bins for continuous variables is the square
            root of the number of samples.  This can be increased or decreased using the "density"
            parameter (see below).  Discrete variables may be binary, categorical or numeric(integer).

            Data(ds) is provided as a dictionary of variable name -> ValueList (List of variable values).
            ValueList should be the same length for all variables.  Each index in ValueList represents
            one sample.

            The density parameter is used to increase or decrease the number of bins used for continuous
            variables.  If density is 1 (default), then sqrt(N-samples) bins are used.  If density is
            set to D, then D * sqrt(N-samples) bins are used.

            The power parameter is used for stochastic approximation of conditional probabilities.
            Power may range from 0 (test conditionality at mean only) up to 100 (test every comination
            of discretized variables).  Values > 0 test more and more points for conditional dependence,
            until at 100, all points are tested.  For linear relationships, power of 0 or 1 is sufficient,
            while for complex, discontinuous relationships, higher values may be necessary to achieve
            high precision.  Power allows a tradeoff between precision and run-time.  High values of
            power may result in unacceptably long run-times.  In practice, power <= 8 should suffice in
            most cases.

            The discSpecs parameter is used to make recursive calls to the module while
            maintaining the discretization information, and should not be provided by the user.

            ProbSpace includes a 'Plot' function that requires matplotlib, and produces a probability
            distribution plot for each variable.

            cMethod determines the methods used for conditional probability of continuous variables.
            - 'u' default uses U-Prob, which automatically balances between D-Prob and J-Prob
            - 'd' forces D-Prob
            - 'j' forces J-Prob

            The main functions of ProbSpace are:
            - P(...) -- Returns the numerical probability of an event, given a set of conditions.
                P can return joint probabilities as well as univariate probabilities.
            - E(...) -- Returns the expected value (i.e. mean) of a variable, given a set of conditions.
            - distr(...) -- Returns a univariate probability distribution (see pdf.py) of a variable
                given a set of conditions.
            - dependence(...) -- Measures the dependence between two variables with optional conditioning
                on a set of other variables (i.e. conditional dependence.).
        """
        assert type(ds) == type({}), "Error -- Data must be in the form of a dictionary varName -> [val1, val2, ... , valN]"
        self.power = power
        self.ds = ds
        self.density = density
        self.fieldList = list(ds.keys())
        self._discreteVars = self._getDiscreteVars()
        self.fieldIndex = {}
        for i in range(len(self.fieldList)):
            key = self.fieldList[i]
            self.fieldIndex[key] = i
        # Convert to Numpy array
        npDat = []
        for field in self.fieldList:
            npDat.append(ds[field])
        self.aData = np.array(npDat)
        self.N = self.aData.shape[1]
        self.probCache = {} # Probability cache
        self.distrCache = {} # Distribution cache
        self.fieldAggs = self.getAgg(ds)
        if self.N:
            if discSpecs:
                self.discSpecs = self.fixupDiscSpecs(discSpecs)
                #self.discretized = self.discretize()
            else:
                self.discSpecs = self.calcDiscSpecs()
                #self.discretized = self.discretize()
        self.cMethod = cMethod
        # For SubSpaces only
        # parentProb is the probability of the subspace within the parent space
        self.parentProb = None
        # Parent Query is the query on the parents space that resulted in the subspace.
        self.parentQuery = None

    def getAgg(self, ds):
        fieldList = list(ds.keys())
        numObs = self.aData.shape[1]  # Number of observations
        if numObs > 0:
            mins = self.aData.min(1)
            maxs = self.aData.max(1)
            means = self.aData.mean(1)
            stds = self.aData.std(1)
        outDict = {}
        for i in range(self.aData.shape[0]):
            fieldName = fieldList[i]
            if numObs:
                aggs = (mins[i], maxs[i], means[i], stds[i])
            else:
                aggs = (0,0,0,0)
            outDict[fieldName] = aggs
        return outDict

    def calcDiscSpecs(self):
        discSpecs = []
        for i in range(len(self.fieldList)):
            var = self.fieldList[i]
            if self.N > 0:
                minV = np.min(self.aData[i])
                maxV = np.max(self.aData[i])
            else:
                minV = 0
                maxV = 0
            isDiscrete = var in self._discreteVars
            if isDiscrete:
                minV = int(minV)
                maxV = int(maxV)
                nBins = maxV - minV + 1
                binStarts = [v for v in range(minV, maxV + 1)]
                vals, counts = np.unique(self.aData[i], return_counts = True)
                hist = np.zeros((len(binStarts),))
                for j in range(len(vals)):
                    val = int(vals[j])
                    count = counts[j]
                    hist[j] = count
                edges = [v for v in range(minV, maxV + 2)]
            else:
                if self.N < 100:
                    nBins = 10
                else:
                    nBins = int(self.density * math.sqrt(self.N))
                hist, edges = np.histogram(self.aData[i], nBins)
            discSpecs.append((nBins, minV, maxV, edges, hist, isDiscrete))
        return discSpecs

    def fixupDiscSpecs(self, discSpecs):
        outSpecs = []
        for i in range(len(discSpecs)):
            discSpec = discSpecs[i]
            bins, min, max, edges, hist, isDiscrete = discSpec
            # Regenerate histogram.  The other data should use the original
            newHist, newEdges = np.histogram(self.aData[i], bins, (min, max))
            outSpecs.append((bins, min, max, edges, newHist, isDiscrete))
        return outSpecs

    # Unused
    def discretize(self):
        discretized = np.copy(self.aData)
        for i in range(len(self.fieldList)):
            field = self.aData[i]
            edges = self.discSpecs[i][3]
            dField = np.digitize(field, edges[:-1]) - 1
            discretized[i] = dField
        return discretized

    def toOriginalForm(self, discretized):
        data = {}
        for f in range(len(self.fieldList)):
            fieldName = self.fieldList[f]
            fieldVals = list(discretized[f, :])
            data[fieldName] = fieldVals
        return data

    # Unused
    def getMidpoints(self, field):
        indx = self.fieldIndex[field]
        dSpec = self.discSpecs[indx]
        edges = dSpec[3]
        isDiscrete = dSpec[5]
        mids = []
        if isDiscrete:
            for i in range(len(edges) - 1):
                mids.append(edges[i])
        else:
            for i in range(len(edges) - 1):
                mids.append((edges[i] + edges[i+1]) / 2)
        return mids

    # Not used
    def getBucketVals(self, field):
        indx = self.fieldIndex[field]
        dSpec = self.discSpecs[indx]
        bucketCount = dSpec[0]
        return range(bucketCount)

    def SubSpace(self, givensSpec, minPoints=None, maxPoints=None, power=None,
                        density=None, discSpecs=None, fixDistr=False):
        """ Return a new ProbSpace object representing a sub-space of the current
            probability space.
            The returned object represents the multivariate joint probability space
            of the original space given a set of conditions.
            That is: P(<all variables> | givensSpec).
            The data is filtered by the givensSpec, and the resulting conditional
            distribution is returned.
            For discrete variables, or filters specified as (varName, high, low),
            exact filtering is done.
            For continuous variables specified as (varName, value), progressive
            filtering is used.  Since the probability of any continuous value is
            zero, the specification (varName, value) is converted to a small
            range (varName, value - delta, value + delta).  This range is iteratively
            adjusted so that a "reasonable" number of data points.  By default,
            that "reasonable" number is between 100 and 1000 data points -- enough
            to produce a reliably measurable distribution.   IF desired, different
            limits can be provided using the minPoints and maxPoints parameters.
            For example, by using minPoints = 1, and maxPoints = 10, one could
            request a small number of samples that are the closest to the requested
            values.  Note that sample counts cannot be managed exactly, so it is
            possible to receieve sample counts smaller or greater than the requested
            range.
        """
        if power is None:
            power = self.power
        if density is None:
            density = self.density
        #print('givens = ', givensSpec)
        #print('minPoints, maxPoints, self.N = ', minPoints, maxPoints, self.N)
        filtDat, parentProb, finalQuery = self.filter(givensSpec, minPoints=minPoints, maxPoints=maxPoints)
        if fixDistr:
            newPS = ProbSpace(filtDat, power = power, density = density, discSpecs = discSpecs, cMethod = self.cMethod)
        else:
            newPS = ProbSpace(filtDat, power = power, density = density, cMethod = self.cMethod)
        newPS.parentProb = parentProb
        newPS.parentQuery = finalQuery
        #print('new N = ', newPS.N)
        return newPS

    def filter(self, filtSpec, minPoints=None, maxPoints=None):
        """ Filter the data based on a set of filterspecs: [filtSpec, ...]
            filtSpec := (varName, value) or (varName, lowValue, highValue)
            Returns a data set in the original {varName:[varData], ...}
            dictionary format.
            See FilteredSpace documentation (above) for details.
        """
        filtdata, parentProb, finalQuery = self.filterDat(filtSpec, minPoints, maxPoints)
        # Now convert to orginal format, with only records that passed filter
        outData = self.toOriginalForm(filtdata)
        return outData, parentProb, finalQuery

    def filterDat(self, filtSpec, minPoints=None, maxPoints=None, adat = None):
        """ Filter the data in its array form and return a filtered array.
            See FilteredSpace documentation (above) for details.
        """
        maxAttempts = 8
        delta = .1
        minPoints_default = max([min([100, sqrt(self.N)]), 20])
        maxPoints_default = max([min([1000, int(self.N / 2)]), minPoints_default * 5])
        if minPoints is None:
            minPoints = minPoints_default
        if maxPoints is None:
            maxPoints = maxPoints_default
        if adat is None:
            adat = self.aData
        
        # Determine if we are doing progressive filtering on at least one variable
        progressive = False
        for filt in filtSpec:
            if len(filt) == 2:
                var, val = filt
                if not self.isDiscrete(var):
                    progressive = True
                    break
        if progressive:
            attempts = maxAttempts
        else:
            attempts = 1
        for attempt in range(attempts):
            # Fix up progressively filtered specs (i.e. non discrete, 2-tuples)
            # to filter on value - delte to value + delta
            filtSpec2 = []
            for filt in filtSpec:
                var = filt[0]
                if self.isDiscrete(var) or len(filt) == 3:
                    filtSpec2.append(filt)
                else:
                    # Progressive
                    val = filt[1]
                    aggs = self.fieldAggs[var]  # Field aggregates
                    std = aggs[3] # Standard deviation
                    sDelta = delta * std # Scaled delta
                    filtSpec2.append((var, val - sDelta, val + sDelta))
            remRecs = []
            for i in range(self.N):
                include = True
                for filt in filtSpec2:
                    if len(filt) == 2:
                        var, targetVal = filt
                        if type(targetVal) == type((0,)):
                            targetVal = val[0]
                        fieldInd = self.fieldIndex[var]
                        val = adat[fieldInd, i]
                        if val != targetVal:
                            include = False
                            break
                    else:
                        var, low, high = filt
                        fieldInd = self.fieldIndex[var]
                        dSpec = self.discSpecs[fieldInd]
                        varMin = dSpec[1]
                        varMax = dSpec[2]
                        if low is None:
                            low = varMin
                        if high is None:
                            high = varMax + .0001
                        val = adat[fieldInd, i]
                        if val < low or val >= high:
                            include = False
                            break
                if not include:
                    remRecs.append(i)
            remaining = self.N - len(remRecs)
            #print('attempt = ', attempt, ', remaining = ', remaining, ', delta = ', delta)
            targetVal = maxPoints * .5
            midpoint = (minPoints + maxPoints) / 2.0
            damping = (1/(attempt+1))
            minFactor = 1.1**damping
            maxFactor = 5**damping
            if remaining > 0:
                ratio = (midpoint / remaining)**damping
            else:
                ratio = maxFactor
            #print('ratio1 = ', ratio)
            #print('minFactor, maxFactor = ', minFactor, maxFactor)
            if ratio >= 1 and ratio < minFactor:
                ratio = minFactor
            elif ratio < 1 and ratio > (1/minFactor):
                ratio = 1/minFactor
            if ratio >= maxFactor:
                ratio = maxFactor
            elif ratio < (1/maxFactor):
                ratio = 1/maxFactor
            #print('ratio2 = ', ratio)
            if remaining < minPoints:
                delta *= ratio
            elif remaining > maxPoints:
                delta *= ratio
            else:
                break
        if DEBUG and progressive:
            print('attempt = ', attempt, ', delta = ', delta, ', remaining = ', remaining, ', minPoints, maxPoints = ', minPoints, maxPoints)
            #print('finalQuery = ', filtSpec2, ', parentProb = ', remaining/self.N, ', parentN = ', self.N)
            pass
        # Remove all the non included rows
        filtered = np.delete(adat, remRecs, 1)
        #print('filtered.shape = ', filtered.shape)
        finalQuery = filtSpec2
        parentProb = filtered.shape[1] / float(self.N)
        return filtered, parentProb, finalQuery
        # End of filterDat

    # Not used
    def binToVal(self, field, bin):
        indx = self.fieldIndex[field]
        dSpec = self.discSpecs[indx]
        min = dSpec[1]
        max = dSpec[2]
        val = (min + max) / 2
        return val

    def makeHashkey(self, targetSpec, givenSpec):
        if type(targetSpec) == type([]):
            targetSpec = tuple(targetSpec)
        if type(givenSpec) == type([]):
            givenSpec = tuple(givenSpec)
        hashKey = (targetSpec, givenSpec)
        return hashKey

    def E(self, target, givensSpec=None, power=None):
        """ Returns the expected value (i.e. mean) of the distribution
            of a single variable given a set of conditions.  This is
            a convenience function equivalent to:
                distr(target, givensSpec).E()

            targetSpec is a single variable name.
            givensSpec is a conditional specification (see distr below
            for format)
        """
        d = self.distr(target, givensSpec, power=power)
        if d.N == 0:
            return None
        return d.E()

    def prob(self, targetSpec, givenSpec=None):
        """ Return the probability of a variable or (set of variables)
            attaining a given value (or range of values) given a set
            of conditionalities on other variables.
            'P' is an alias for prob.
            The basic form is the probability of Y given X or the probability
            of targetSpec given givenSpec, where X and Y represent events or lists
            of simultaneous events:
                P(Y=y | X=x) = P(targetSpec | givenSpec)

            targetSpec (target specification) defines the result to be returned
            (i.e. the event or set of events whose probability is to be determined).
            A target specification may take one of several forms:
            - 2-tuple (varName, value) for the probability
                of attaining a single value
            - 3-tuple: (varName, minValue, maxValue) indicating an interval:
                [minValue, maxValue) (i.e. minValue <= value < maxValue).
                minValue or maxValue may be None.  A minValue of None implies 
                -infinity.  A maxValue of None implies infinity.
            - list of either of the above or any combination.  In this case, the joint 
                probability of the events is returned.
            
            givenSpec (optional) has the same format as targetSpec with equivalent meanings
            for the givens.  A given specification supports one additional flavor which
            is a single variable name.  This means that that variable should be
            "conditionalized" on.  
            The three flavors (varName, 2-tuple, 3-tuple)
            may be mixed within a givenSpec, presented as a list of givens.

            Examples:
            - prob(('A', 1)) -- The probability that variable A takes on the value 1.
            - prob([('A', 1), ('B', 2)]) -- The (joint) probability that A is 1 and B is 2.
            - prob(('A', .1, .5)) -- The probability that A is in the range [.1, .5).
            - prob(('A', .1, .5), [('B', 0, 1), ('C', -1, None)] -- The probability that
                    A is on interval [.1, .5) given that B is on interval [0,1) and
                    C is on interval [-1, infinity).
            - prob(('A', .1, .5), [('B', 0, 1), ('C', -1, None), 'D'] -- The probability that
                variable A is on interval [.1, .5) given that B is on interval [0, 1) and
                    C is on interval [-1, infinity), conditionalized on D.
    
            Conditionalization is taking the probability weighted sum of the results for every value
            of the conditionalizing variable or combination of conditionalizing variables.
            For example:
            - P(A=1 | B=2, C) is: sum over all (C=c) values( P(A=1 | B=2, C=c) * P(C=c))
        """
        if givenSpec is None:
            givenSpec = []
        if type(givenSpec) != type([]):
            givenSpec = [givenSpec]
        cacheKey = self.makeHashkey(targetSpec, givenSpec)
        if cacheKey in self.probCache.keys():
            return self.probCache[cacheKey]
        if type(targetSpec) == type([]):
            # Joint probability
            # Separate unbound (e.g. A) specifications from bound (e.g. A=1) specifications
            # Prob doesn't return unbound results.
            targetSpecU, targetSpec = self.separateSpecs(targetSpec)
            assert len(targetSpecU) == 0, 'prob.P: All target specifications must be bound (i.e. specified as tuples).  For unbound returns, use distr.)'
            result = self.jointProb(targetSpec, givenSpec)
            self.probCache[cacheKey] = result
        else:
            rvName = targetSpec[0]
            if len(targetSpec) == 2:
                valSpec = targetSpec[1]
            else:
                valSpec = targetSpec[1:]
            d = self.distr(rvName, givenSpec)
            if d.N > 0:
                result = d.P(valSpec)
            else:
                result = None
            self.probCache[cacheKey] = result
        return result

    P = prob


    def distr(self, rvName, givenSpecs=None, power=None):
        """Return a univariate probability distribution as a PDF (see pdf.py) for the random variable
           indicated by rvName.
           If givenSpec is provided, then will return the conditional distribution,
           otherwise will return the unconditional (i.e. marginal) distribution.
           This satisfies the following types of probability queries:
            - P(Y) -- (marginal) Probability distribution of Y
            - P(Y | X=x) -- Conditional probability
            - P(Y | X1=x1, ... ,Xk = xk) -- Multiple conditions
            - P(Y | X=x, Z) -- i.e. Conditionalize on Z
            - P(Y | X=x, Z1, ... Zk) -- Conditionalize on multiple variables

            rvName is the name of the random variable whose distribution is requested.
            givenSpec (given specification) defines the conditions (givens) to
            be applied.
            A given specification may take one of several forms:
            - 2-tuple (varName, value) - Variable taking on a given value.
            - 3-tuple: (varName, minValue, maxValue) indicating an interval:
                [minValue, maxValue) (i.e. minValue <= value < maxValue).
                minValue or maxValue may be None.  A minValue of None implies 
                -infinity.  A maxValue of None implies infinity.
            - variable name: A variable to conditionalize on.
            - list of any of the above or any combination of above.

            Examples:
            - distr('Y') -- The (marginal) probability of Y
            - distr('Y', [('X', 1)]) -- The probability of Y given X=1.
            - distr('Y', [('X', 1, 2)]) -- The probability of Y given 1 <= X < 2.
            - distr('Y', ('X', 1)) -- The probability of Y given X=1 (same as above)
            - distr('Y', [('X1', 1), ('X2', 0)]) - The probability of Y given X1 = 1, and X2 = 0
            - distr('Y', [('X', 1), 'Z']) -- The probability of Y given X = 1, conditionalized on Z

            Conditionalization is taking the probability weighted sum of the results for every value
            of the conditionalizing variable or combination of conditionalizing variables.
            For example:
            - P(Y | X=1, Z) is: sum over all (Z=z) values( P(Y | X=1, Z=z) * P(Z=z))
        """
        DISC_DISTS = False # If False, use full data for distributions.  Otherwise use binned data.
                          # Will be much slower and slightly more accurate if False.

        if power is None:
            power = self.power
        if givenSpecs is None:
            givenSpecs = []
        if type(givenSpecs) != type([]):
            givenSpecs = [givenSpecs]
        cacheKey = self.makeHashkey(rvName, givenSpecs)
        if cacheKey in self.distrCache.keys():
             return self.distrCache[cacheKey]
        if DEBUG:
            print('ProbSpace.distr: P(' , rvName, '|', givenSpecs , ')')
        isDiscrete = self.isDiscrete(rvName)
        indx = self.fieldIndex[rvName]
        dSpec = self.discSpecs[indx]
        bins = dSpec[0]

        if not givenSpecs:
            # Marginal (unconditional) Probability
            bins = dSpec[0]
            hist = list(dSpec[4])
            if not hist:
                hist = [0] * bins
            edges = list(dSpec[3])
            outHist = []
            for i in range(len(hist)):
                cnt = hist[i]
                if self.N > 0:
                    outHist.append(cnt / self.N)
                else:
                    outHist.append(0)
            pdfSpec = []
            for i in range(len(outHist)):
                start = edges[i]
                end = edges[i+1]
                pdfSpec.append((i, start, end, outHist[i]))
            if not DISC_DISTS:
                dat = self.aData[indx,:]
            else:
                dat = None
            outPDF = PDF(self.N, pdfSpec, isDiscrete=isDiscrete, data=dat)
        else:
            # Conditional Probability
            condSpecs, filtSpecs = self.separateSpecs(givenSpecs)
            if not condSpecs:
                # Nothing to conditionalize on.  We're computing a fully bound conditional (i.e. no free variables)
                outPDF = self.boundCondition(rvName, filtSpecs)
            else:
                # Conditionalize on all indicated variables. I.e.,
                # SUM(P(filteredY | Z=z) * P(Z=z)) for all z in Z.
                accum = np.zeros((bins,))
                conditionalizeOn = []
                for given in condSpecs:
                    conditionalizeOn.append(given)
                totalDepth = len(conditionalizeOn)
                if filtSpecs:
                    totalDepth += len(filtSpecs)
                #print('totalDepth = ', totalDepth)
                p = self.reductionExponent(totalDepth)
                #print('p = ', p)
                #condFiltSpecs = self.getCondSpecs(conditionalizeOn, power=power, hierarchical=True, reductionExp=p)
                #countRatio = float(self.N) / filtSample.N
                allProbs = 0.0 # The fraction of the probability space that has been tested.

                Dtarg = 1 # The target dim.
                Dfilt = len(filtSpecs)  # Filterning Dom
                Dcond = len(condSpecs)  # Conditionalize Dim
                Dquery = Dtarg + Dfilt + Dcond # Query Dim
                Nfilt = self.N**((Dtarg + Dcond)/Dquery) # Number of points to return from filter
                # First, we filter on the bound conditions (if any), then conditionalize on the reduced set
                if filtSpecs:
                    minP_Filt = .8 * Nfilt
                    maxP_Filt = 1.2 * Nfilt
                    ss = self.SubSpace(filtSpecs, minPoints=minP_Filt, maxPoints=maxP_Filt, discSpecs=self.discSpecs, fixDistr=True)
                else:
                    ss = self
                Ntarg = self.N**(Dtarg/(Dcond + Dtarg)) # Number of points to return from final
                minP = .8 * Ntarg
                maxP = 1.2 * Ntarg
                condFiltSpecs = self.getCondSpecs(conditionalizeOn, power=power, hierarchical=False, reductionExp=p)
                for cf in condFiltSpecs:
                    # Create a new subspace filtered by both the bound and unbound conditions
                    # Note that progressive filtering will be used for the unbound conditions.
                    # probYgZ is P(Y | Z=z) e.g., P(Y | X=1, Z=z)
                    #print('N, min, max = ', self.N, minPoints, maxPoints)
                    ss2 = ss.SubSpace(cf, minPoints=minP, maxPoints=maxP, discSpecs=self.discSpecs, fixDistr=True)
                    #print('ss2.N = ', ss2.N)
                    probYgZ = ss2.distr(rvName)
                    #probYgZ = filtSpace.distr(rvName, cf)
                    # Now we can compute probZ as ratio of the number of data points in the filtered distribution and the original
                    probZ = ss2.N / ss.N
                    #print('probZ = ', probZ, ', probYgZ/E() = ', probYgZ.E(), ', probYgZ.N = ', probYgZ.N, ', ss.N = ', ss.N, ', ss.query = ', ss.parentQuery, ', ss2.query = ', ss2.parentQuery)
                    if probZ == 0:
                        # Zero probability -- don't bother accumulating
                        continue
                    probs = probYgZ.ToHistogram() * probZ # Creates an array of probabilities
                    accum += probs
                    allProbs += probZ
                accum = accum / allProbs
                # Now we start with a pdf of the original variable to establish the ranges, and
                # then replace the actual probabilities of each bin.  That way we maintain the
                # original bin structure. 
                template = self.distr(rvName)
                outSpecs = []
                for i in range(len(accum)):
                    pdfBin = template.getBin(i)
                    newprob = accum[i]
                    newBin = pdfBin[:-1] + (newprob,)
                    outSpecs.append(newBin)
                outPDF = PDF(ss2.N, outSpecs, isDiscrete = isDiscrete)
        self.distrCache[cacheKey] = outPDF
        return outPDF

    PD = distr

    # Return (targIsDiscrete, discConds, contConds)
    def analyzeQuery(self, rvName, condSpecs):
        targIsDisc = self.isDiscrete(rvName) # Is Target Discrete
        discConds = [] # Discrete conditions
        contConds = [] # Continuous conditions
        for condSpec in condSpecs:
            varName = condSpec[0]

            if self.isDiscrete(varName):
                discConds.append(condSpec)
            else:
                contConds.append(condSpec)
        return (targIsDisc, discConds, contConds)
        

    def boundCondition(self, rvName, condSpecs):
        #print('***** Bound Condition')
        targIsDisc, discConds, contConds = self.analyzeQuery(rvName, condSpecs)
        
        if targIsDisc:
            # Use only filtering if target is discrete.  We need a distribution for the
            # target in order to analyze.  

            filtConds = condSpecs
            contConds = []
        else:
            filtConds = discConds

        if filtConds:
            # We have variables that need filtering (either due to discrete conditionals, or discrete target).
            # Create a new subspace with the filter variables removed.
            Dtarg = len(contConds) + 1 # Dimension of continous conditions plus 1 for the target.
            Dcond = len(filtConds)  # Number of dimensions to filter
            Dquery = Dtarg + Dcond # Dim of the query
            Ntarg = self.N**(Dtarg/Dquery) # Number of points to return from filter
            filtSpace = self.SubSpace(filtConds, density = self.density, power = self.power, discSpecs=self.discSpecs, minPoints = Ntarg * .8, maxPoints = Ntarg * 1.2)
        else:
            filtSpace = self  # No required pre-filtering

        #filtSpace = self.SubSpace(condSpecs, density = self.density, power = self.power, minPoints = 100, maxPoints = sqrt(self.N))
        if contConds:
            # We still have continuous variables to process. Use U-Prob(L)
            # Set L, the Lambda value for U-Prob
            if self.cMethod == 'd':
                L = 100
            elif self.cMethod == 'j':
                L = 0
            else:
                # Let U-Prob utomatically determine setting for L
                L = None
            if self.cMethod != 'd!':  # Anything but old stle discretiza\ation
                # Use the filtered space remaining after removing discrete vars.
                upr = uprob.UPROB(filtSpace, rvName, condSpecs=contConds, lmbda=L)
                outPDF = upr.distr()
            else:
                # d! -- forced discretization -- old style
                Dtarg = 1 # The target dim.
                Dcond = len(contConds)  # Conditional Dim
                Dquery = Dtarg + Dcond # Query Dim
                Ntarg = filtSpace.N**(Dtarg/Dquery) # Number of points to return from filter
                filtSpace = filtSpace.SubSpace(contConds, density = self.density, power = self.power, discSpecs=self.discSpecs, minPoints = Ntarg * .8, maxPoints = Ntarg * 1.2)
                if filtSpace.N > 0:
                    #print('filtspace.N, parentQuery = ', filtSpace.N, filtSpace.parentQuery)
                    outPDF = filtSpace.distr(rvName)
                else:
                    outPDF = None
        else:
            # We have already processed all conditions.  We have the final distribution.
            pass

            if filtSpace.N > 0:
                #print('filtspace.N, parentQuery = ', filtSpace.N, filtSpace.parentQuery)
                outPDF = filtSpace.distr(rvName)
            else:
                outPDF = None
        return outPDF

    def reductionExponent(self, totalDepth):
        minPoints = 200
        p = log(minPoints, self.N)**(1 / totalDepth)
        return p

    def getCondSpecs(self, condVars, power=2, hierarchical=True, reductionExp=None):
        """ Produce a set of conditional specifications for stochastic
            conditionalization, given
            a set of variables to conditionalize on, and a desired power level.
            Power determines how many points to use to conditionalize on.
            Zero indicates conditionalize on the mean alone. 1 uses the mean
            and two other points (one on either side of the mean).
            2 Uses the mean plus 4 other points (2 on each side of the mean).
            Power values (p) less than 100 will test p * 2 + 1 values for each
            variable.range
            power value > 100 indicates that all values will be tested,
            which can be extremely processor intensive.
            Conditional specifications provide a list of lists of tuple:
            [[(varName1, value1_1), (varName2, value2_1), ... (varNameK, valueK_1)],
             [(varName1, value1_2), (varName2, value2_2), ... (varNameK, valueK_2)],
             ...
             [(varName1, value1_N), (varName2, value2_N), ... (varNameK, valueK_N)]]
            Where K is the number of conditional variables, and N is the total number
            of combinations = K**(2 * P + 1) for values of P < 100.
        """
        rawCS = self.getCondSpecs2(condVars, power = power)
        if reductionExp is None:
            p = self.reductionExponent(len(condVars)) # The exponent of the number of data points for each level
        else:
            p = reductionExp
        #print('p = ', p)
        #print('rawCS = ', rawCS, ', power = ', power)
        outCS = []
        rotation = 0
        for spec in rawCS:
            # If hierarchical sampling, rotate the variables so that they take turns being
            # the top variable.
            if rotation > 0 and hierarchical:
                spec = spec[rotation:] + spec[:rotation]
            #print('spec = ', spec)
            currPS = self
            outSpec = []
            # Adjust pseudo filters by the mean and std of the conditional
            for s in range(len(spec)):
                varSpec = spec[s]
                #print('varSpec = ', varSpec)
                var = varSpec[0]
                #print('var = ', var)
                if self.isDiscrete(var):
                    outSpec.append(varSpec)
                else:
                    val = varSpec[1]
                    distr = currPS.distr(var)
                    mean = distr.E()
                    std = distr.stDev()
                    #print('mean, std = ', mean, std)
                    varSpec = (var, mean + val * std)
                    #print('varSpec = ', varSpec, ', val = ', val, ', mean, std = ', mean, std)
                    outSpec.append(varSpec)
                # Use resulting conditional space of this variable for the sample
                # of the next one.  That way, we sample using the mean and std
                # of the variable in the conditioned space of the previous vars.
                if s != len(spec) - 1 and hierarchical:
                    dQuery = len(spec)
                    nTarg = self.N**((s+1)/dQuery)
                    minPoints = .8 * nTarg
                    maxPoints = 1.2 * nTarg
                    #print('N, minPoints, maxPoints = ', currPS.N, minPoints, maxPoints)
                    currPS = currPS.SubSpace([varSpec], minPoints=minPoints, maxPoints=maxPoints)
            # Only need to rotate if doing hierarchical
            if hierarchical:
                rotation = (rotation + 1) % len(condVars)
            #print('outSpec = ', outSpec)
            outCS.append(outSpec)
        #print('outCS = ', outCS)
        return outCS

    def getCondSpecs2(self, condVars, power=2):
        def getTestVals(self, rv):
            isDiscrete = self.isDiscrete(rvName)
            if isDiscrete or testPoints is None:
                # If is Discrete, return all values
                testVals = self.getMidpoints(rv)
                testVals = [testVal for testVal in testVals]
            else:
                # If continuous, sample values at testPoint distances from the mean
                testVals = []
                for tp in testPoints:
                    if tp == 0:
                        # For 0, just use the mean
                        testVals.append(0)
                    else:
                        # For nonzero, test points mean + tp and mean - tp
                        testVals.append(-tp,)
                        testVals.append(tp,)
            return testVals
        levelSpecs0 = [.5, .25, .75, 1, .25, .1, 1.5, 1.25, 1.75, 2.0]
        maxLevel = 3 # Largest standard deviation to sample
        if power <= 10:
            levelSpecs = levelSpecs0
        elif power < 100:
            levelSpecs = list(np.arange(1/power, maxLevel + 1/power, 1/power))
        else:
            levelSpecs = None
        if levelSpecs:
            testPoints = [0] + levelSpecs[:power]
        else:
            # TestPoints None means test all values
            testPoints = None

        # Find values for each variable based on testPoints
        nVars = len(condVars)
        rvName = condVars[0]
        if nVars == 1:
            # Only one var to do.  Find the values.
            vals = getTestVals(self, rvName)
            return [[(rvName, val)] for val in vals]
        else:
            # We're not on the last var, so recurse and build up the total set
            accum = []
            vals = getTestVals(self, rvName)
            nextPower = power if power <= 1 else power -1
            childVals = self.getCondSpecs2(condVars[1:], nextPower) # Recurse to get the child values
            for val in vals:
                accum += [[(rvName, val)] + childVal for childVal in childVals]
            return accum
        # End of getCondSpecs
        

    def adjustSpec(self, spec, delta):
        outSpec = []
        for var in spec:
            if len(var) == 2:
                # Discrete.  Don't modify
                outSpec.append(var)
            else:
                varName, low, high = var
                mid = (low + high) / 2.0
                oldDelta = (high - low) / 2.0
                newDelta = delta * oldDelta
                newVar = (varName, mid - newDelta, mid + newDelta)
                #print('oldVar = ', var, 'newVar = ', newVar, mid, oldDelta, newDelta, deltaAdjust)
                outSpec.append(newVar)
        #print('old = ', spec, ', new =', outSpec, ', delta = ', deltaAdjust)
        return outSpec

    def dependence(self, rv1, rv2, givensSpec=[], power=None, raw=False):
        """ givens is [given1, given2, ... , givenN]
        """
        if power is None:
            power = self.power
        if givensSpec is None:
            givensSpec = []
        if type(givensSpec) != type([]):
            givensSpec = [givensSpec]
        accum = 0.0
        accumProb = 0.0
        # Get all the combinations of rv1, rv2, and any givens
        # Depending on power, we test more combinations.  If level >= 100, we test all combos
        # For level = 0, we just test the mean.  For 1, we test the mean + 2 more values.
        # For level = 3, we test the mean + 6 more values.

        # Separate the givens into bound (e.g. B=1, 1 <= B < 2) and unbound (e.g., B) specifications.
        totalDepth = len(givensSpec) + 1
        p = self.reductionExponent(totalDepth)
        #print('p =', p)
        givensU, givensB = self.separateSpecs(givensSpec)
        if not givensU:
            condFiltSpecs = [None]
        else:
            condFiltSpecs = self.getCondSpecs(givensU, power, reductionExp=p)    
        prevGivens = None
        prevProb1 = None
        numTests = 0
        for spec in condFiltSpecs:
            # Compare P(Y | Z) with P(Y | X,Z)
            # givens is conditional on spec without rv2
            #print('spec = ', spec)
            if spec is None: # Unconditional Independence
                minPoints = self.N**(p**len(givensB))
                maxPoints = minPoints * 5
                ss1 = self.SubSpace(givensB, minPoints=minPoints, maxPoints=maxPoints)
                prob1 = ss1.distr(rv1)
            else:
                givens = spec
                if givens != prevGivens:
                    # Only recompute prob 1 when givensValues change
                    # Get a subsapce filtered by all givens, but not rv2
                    minPoints = self.N**(p**(len(givensB)+len(givens)))
                    maxPoints = minPoints * 5
                    ss1 = self.SubSpace(givens + givensB, minPoints=minPoints, maxPoints=maxPoints)
                    prob1 = ss1.distr(rv1)
                    prevProb1 = prob1
                    prevGivens = givens
                else:
                    # Otherwise use the previously computed prob1
                    prob1 = prevProb1
            if prob1.N == 0:
                print('Empty distribution: ', spec)
                continue
            testSpecs = ss1.getCondSpecs([rv2], power)
            for testSpec in testSpecs:
                # prob2 is the conditional subspace of everything but rv2
                # conditioned on rv2
                minPoints = ss1.N**(p)
                maxPoints = minPoints * 5
                ss2 = ss1.SubSpace(testSpec, minPoints=minPoints, maxPoints=maxPoints)
                prob2 = ss2.distr(rv1)
                if prob2.N == 0:
                    continue
                dep = prob1.compare(prob2, raw=raw)
                # We accumulate any measured dependency multiplied by the probability of the conditional
                # clause.  This way, we weight the dependency by the frequency of the event.
                condProb = prob2.N / self.N
                accum += dep * condProb
                accumProb += condProb # The total probability space assessed
                numTests += 1
                if DEBUG:
                    print('spec = ', spec, testSpec, ', givensB = ', givensB, ', dep = ', dep, ', prob1.N, prob2.N = ', prob1.N, prob2.N)
                    print('ss1.parentQuery = ', ss1.parentQuery, ', ss2.parentQuery = ', ss2.parentQuery)
            #print('ss1.N, ss2.N = ', ss1.N, ss2.N)
        if accumProb > 0.0:
            # Normalize the results for the probability space sampled by dividing by accumProb
            dependence = accum / accumProb
            if not raw:
                # Bound it to [0, 1] 
                dependence = max([min([dependence, 1]), 0])
            return dependence
            #H = .36845
            #L = .0272
            #if dependence < L:
            #    calDep = dependence / (2*L)
            #else:
            #    calDep = (dependence - L) / (H-L) / 2 + .5
            # Bound it to [0, 1] 
            #calDep = max([min([calDep, 1]), 0])
            #return calDep

        print('Cond distr too small: ', rv1, rv2, givens)
        return 0.0

    def separateSpecs(self, specs):
        """ Separate bound and unbound variable specs,
            and return (unboundSpecs, boundSpecs).  
        """
        delta = .05
        uSpecs = []
        bSpecs = []
        for spec in specs:
            if type(spec) == type((0,)):
                # It is a bound spec
                bSpecs.append(spec)
            else:
                # Unbound
                uSpecs.append(spec)
        return uSpecs, bSpecs



    def independence(self, rv1, rv2, givensSpec=None, power=None):
        """
            Calculate the independence between two variables, and an optional set of givens.
            This is a heuristic inversion
            of the dependence calculation to match other independence measures which return
            the likelihood of the null hypothesis that the variables are dependent.
            A threshold of .1 is generally used.  Values below that are considered dependent.
            givens are formatted the same as for prob(...).
            TO DO: Calibrate to an exact p-value.
        """
        dep = self.dependence(rv1, rv2, givensSpec=givensSpec, power=power)
        ind = 1 - dep
        return ind


    def isIndependent(self, rv1, rv2, givensSpec=None, power=None):
        """ Determines if two variables are independent, optionally given a set of givens.
            Returns True if independent, otherwise False
        """
        ind = self.independence(rv1, rv2, givensSpec = givensSpec, power = power)
        # Use .5 (50% confidence as threshold.
        return ind > .5

    # Unused
    def jointValues(self, rvList):
        """ Return a list of the joint distribution values for a set of variables.
            I.e. [(rv1Val, rv2Val, ... , rvNVal)] for every combination of bin values.
        """
        nVars = len(rvList)
        rvName = rvList[0]
        vals = self.getMidpoints(rvName)
        if nVars == 1:
            return ([(val,) for val in vals])
        else:
            accum = []
            childVals = self.jointValues(rvList[1:]) # Recurse to get the child values
            for val in vals:
                accum += [(val,) + childVal for childVal in childVals]
            return accum

    # Unused
    def jointCondSpecs(self, rvList):
        condSpecList = []
        jointVals = self.jointValues(rvList)
        for item in jointVals:
            condSpecs = []
            for i in range(len(rvList)):
                rvName = rvList[i]
                val = item[i]
                spec = (rvName, val)
                condSpecs.append(spec)
            condSpecList.append(condSpecs)
        return condSpecList

    def jointProb(self, varSpecs, givenSpecs=None):
        """ Return the joint probability given a set of variables and their
            values.  varSpecs is of the form (varName, varVal).  We want
            to find the probability of all of the named variables having
            the designated value.
            Join Probability is calculated as e.g.:
            - P(A, B, C) = P(A | B,C) * P(B | C) * P(C)
        """
        if givenSpecs is None:
            givenSpecs = []
        accum = []
        nSpecs = len(varSpecs)
        for i in range(nSpecs):
            spec = varSpecs[i]
            if i == nSpecs - 1:
                accum.append(self.prob(spec, givenSpecs))
            else:
                nextSpecs = varSpecs[i+1:]
                accum.append(self.prob(spec, nextSpecs + givenSpecs))

        # Return the product of the accumulated probabilities
        allProbs = np.array(accum)
        jointProb = float(np.prod(allProbs))
        return jointProb

    def pdfToProbArray(self, pdf):
        vals = []
        for bin in pdf:
            prob = bin[3]
            vals.append(prob)
        outArray = np.array(vals)
        return outArray

    def fieldStats(self, field):
        return self.fieldAggs[field]

    def _isDiscreteVar(self, rvName):
        vals = self.ds[rvName]
        isDiscrete = True
        for val in vals:
            if val != int(val):
                isDiscrete = False
                break
        return isDiscrete
    
    def _getDiscreteVars(self):
        discreteVars = []
        for var in self.fieldList:
            if self._isDiscreteVar(var):
                discreteVars.append(var)
        return discreteVars

    def isDiscrete(self, rvName):
        indx = self.fieldIndex[rvName]
        dSpec = self.discSpecs[indx]
        isDisc = dSpec[5]
        return isDisc

    def corrCoef(self, rv1, rv2):
        """Pearson Correlation Coefficient (rho)
        """
        indx1 = self.fieldIndex[rv1]
        indx2 = self.fieldIndex[rv2]
        dat1 = self.aData[indx1,:]
        dat2 = self.aData[indx2,:]
        mean1 = dat1.mean()
        mean2 = dat2.mean()
        num1 = 0.0
        denom1 = 0.0
        denom2 = 0.0
        for i in range(self.N):
            v1 = dat1[i]
            v2 = dat2[i]
            diff1 = v1 - mean1
            diff2 = v2 - mean2
            num1 += diff1 * diff2
            denom1 += diff1**2
            denom2 += diff2**2
        rho = num1 / (denom1**.5 * denom2**.5)
        return rho

    def Predict(self, Y, X, useVars=None):
        """
            Y is a single variable name.  X is a dataset.
        """
        dists = self.PredictDist(Y, X, useVars)
        preds = [dist.E() for dist in dists]
        return preds

    def Classify(self, Y, X, useVars=None):
        """
            Y is a single variable name.  X is a dataset.
        """
        assert self.isDiscrete(Y), 'Prob.Classify: Target variable must be discrete.'
        dists = self.PredictDist(Y, X, useVars)
        preds = [dist.mode() for dist in dists]
        return preds

    def PredictDist(self, Y, X, useVars=None):
        """
            Y is a single variable name.  X is a dataset.
        """
        outPreds = []
        # Make sure Y is not in X
        if useVars is not None:
            # Use the independent variables as specified
            vars = useVars
        else:
            # Calculate the variables to use.  No point
            # in using variables that are independent of the target,
            # so we filter those out.  We also make sure the target is
            # not in the variable list.
            vars = list(X.keys())
            # Make sure the target is not in the variable list.
            try:
                vars.remove(Y)
            except:
                pass
            # Sort the independent variables by dependence with Y
            deps = [(self.dependence(var, Y, power=3), var) for var in vars]
            deps.sort()
            deps.reverse()
            vars = []
            # Remove any independent independents
            for i in range(len(deps)):
                dep = deps[i]
                if dep[0] < .5:
                #if dep[0] < 0: # Temp disable
                    print('Prob.PredictDist: rejecting variables due to independence from target(p-value, var): ', deps[i:])
                    break
                else:
                    vars.append(dep[1])
            #print('vars = ', vars)
            # Vars now contains all the variables that are not indpendent from Y, sorted in
            # order of highest dependence.            

        # Get the number of items to predict:
        numTests = len(X[vars[0]])
        targetIsDiscrete = self.isDiscrete(Y)
        for i in range(numTests):
            filts = []
            for var in vars:
                val = X[var][i]
                filts.append((var, val))
            maxPoints = min([sqrt(self.N), 20])
            fs = self.SubSpace(filts, minPoints=1, maxPoints=maxPoints)
            #fs = self.SubSpace(filts)
            if DEBUG and fs.N > maxPoints and not targetIsDiscrete:
                print('subspace.N = ', fs.N)
            if fs.N == 0:
                if DEBUG:
                    print('no examples found.')
                pred = self.distr(Y) # Use the natural target distribution
            else:
                pred = fs.distr(Y)
            outPreds.append(pred)
        return outPreds

    def Plot(self):
        """ Plot the distribution of each variable in the joint probability space
            using matplotlib.
        """
        inf = 10**30
        plotDict = {}
        minX = inf
        maxX = -inf
        numPts = 200
        pdfs = []
        for v in self.fieldList:
            d = self.distr(v)
            minval = d.minVal()
            maxval = d.maxVal()
            if maxval > maxX:
                maxX = maxval
            if minval < minX:
                minX = minval
            pdfs.append(d)
        xvals = []
        for i in range(numPts):
            rangex = maxX - minX
            incrx = rangex / numPts
            xvals.append(minX + i * incrx)
        for i in range(len(self.fieldList)):
            yvals = []
            var = self.fieldList[i]
            pdf = pdfs[i]
            for j in range(numPts):
                xval = xvals[j]
                if j == numPts - 1:
                    P = pdf.P((xval, maxX))
                else:
                    P = pdf.P((xval, xvals[j+1]))
                yvals.append(P)
            plotDict[var] = yvals
        plotDict['_x_'] = xvals
        probCharts.plot(plotDict)

