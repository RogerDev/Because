""" Multivariate RKHS.  
    @param data is a dictionary of {<variable name> -> array-like list of data points 1-N}
    @param includeVars is the list of variables from which to build the multivariate joint
    distribution. That allows the same data set to be passed around with different 
    includeVars to generate different expressions.
    @param s is a smoothness factor.  The default is 1.0. Increasing this factor increases
    the smoothness of the model.  Decreasing allows more details, with less smoothing.   
    @param delta -- Not used at this time.
"""
from math import e, sqrt, pi, log, gamma
import numpy as np
from scipy import stats

class RKHS:
    def __init__(self, data, includeVars = None, delta = 3, s=1.0):
        """
        """
        #assert type(data) == type({}), "rkhsMV: Error -- Data must be in the form of a dictionary varName -> [val1, val2, ... , valN]"
        self.data = data
        # If includeVars is present, use it to build the samples and associated sigmas
        self.vMap = {} # Map from var name to var index
        self.iMap = {} # Map from index to varName
        self.s = s # Smoothness factor
        if includeVars is None:
            self.varNames = list(data.keys())
        else:
            self.varNames = includeVars
        self.D = len(self.varNames)
        self.N = len(data[self.varNames[0]])
        # X holds our data samples, reorganized as an array [D x N]
        self.X = []
        tempA = []
        self.stdDevs = []
        for i in range(self.D):
            varName = self.varNames[i]
            dat = data[varName]
            stdDev = np.std(dat)
            self.stdDevs.append(stdDev)
            self.vMap[varName] = i
            self.iMap[i] = varName
            tempA.append(dat)
        for i in range(self.N):
            sample = []
            for j in range(self.D):
                val = tempA[j][i]
                sample.append(val)
            self.X.append(sample)
        self.X = np.array(self.X)
        # Compute covariance matrix for kernel
        Xa = np.array(self.X).T
        c = np.cov(Xa).reshape(self.D, self.D)
        #print('c = ', c)
        # Make sure we scale the standard deviations
        # and not the variances.  The diagonal of the
        # covariance matrix contains the variances of
        # each variable.
        s2 = np.diag(c)
        self.stds = []
        for sig2 in s2:
            sig = sqrt(sig2)
            self.stds.append(sig)
        # We scale the stds of the original data to get the stds of
        # the mv Gaussians we place at each data point.
        # Scale encompasses 3 variables:
        #   N - the number of data points
        #   D - the number of variables
        #   S - the data specific smoothness (optionally) provided
        #           by the user
        # We         
        scale = (.5**(1/self.D) * s * self.N**(-1/5))
        #print('rkhsMV: scale = ', scale)
        #scale = (.5 * s * self.N**(-1/6))
        cS = np.zeros(shape=c.shape)
        #print('c = ')
        for i in range(self.D):
            for j in range(self.D):
                v = c[i,j]
                cS[i,j] = (sqrt(v) * scale)**2 if v >=0 else -(sqrt(-v) * scale)**2
                #cS[i,j] = v * scale
        #print('c = ', c, c.shape)
        #cS = c * scale
        #print('cs = ', cS, c.shape)
        #5/0
        self.delta = delta
        # Calculate the covariance matrix, assuming all vars are independent
        self.covM = cS
        # Precompute inverse covariance matrix
        self.covMI = np.linalg.inv(self.covM)
        #print('covMI = ', self.covMI)
        # Precompute determinant of covM
        self.detcov = np.linalg.det(self.covM)
        # Precompute the denominator for the kernel function
        self.denom = sqrt((2*pi)**(self.D) * self.detcov) * 2
        #print('self.N = ', self.N)
        #print('self.D = ', self.D)
        # Precompute range variables for efficency
        self.nrange = range(self.N)
        self.drange = range(self.D)
        self.mv = stats.multivariate_normal(mean=[0]*self.D, cov=self.covM)
        # Secondary RKHS for conditioning (cache)
        self.R2 = None

    
    def K(self, x1, x2=0):
        """ Multivariate gaussian density kernel
            x1 and x2 are vectors of length D.
            returns a density vector of length D.
        """
        diffs = (x1 - x2).reshape(self.D, 1)  # Vector of differences
        #diffs = np.array([0] * self.D).reshape(self.D, 1)
        return e**(-.5 * (diffs.transpose() @ self.covMI @ diffs)[0][0]) / self.denom
        #return density


    def P(self, x):
        """ 
            Produces Joint Probability = P(X1=x1, ... , XN=xN).
            x is an array of x values, one for
            each dimension of the joint probability.
        """
        # Evaluate the MV Gaussian pdf kernel
        v = 0.0
        for p in self.X:
            p = np.array(p)
            v += self.K(p, x)
        result = v / self.N
        return result
 
    def CDF(self, x):
        """ 
            Produces Cumulative Joint Distribution:
                CDF(X1=x1, X2=x2, ..., XN=xN)

        """
        # Evaluate the MV Gaussian cdf kernel
        v = 0.0
        mv = self.mv
        for i in self.nrange:
            p = self.X[i]
            p = np.array(p)
            c = mv.cdf(x-p)
            v += c
        return v / self.N

    def condP(self, x):
        """ Produces Conditional Probability:
                P(X1=x1 | X2=x2, ... XN=xN)
            The probability density of first variable
            in includeVars given the other variables is
            returned.
        """
        # Calculate Joint Probability (JP) of X1-XN / JP of X2-XN)
        # i.e JP(X1, ..., XN) / JP(X2, ... , XN)

        # Create or use a separate RKHS that has includeVars = 
        # our includeVars 2-N.
        if self.R2 is None:
            condVars = self.varNames[1:]
            self.R2 = RKHS(self.data, condVars, s=self.s)
        R2 = self.R2
        v1 = self.P(x)
        # Strip the first value (x1) from our x and use for
        # second RKHS.
        v2 = R2.P(x[1:])
        if v2 > 0:
            result = v1/v2
            if result > 1.0:
                print('rkhsMV.condP: P > 1??  v1, v2 = ', v1, v2)
        else:
            result = None
        return result

    def condE(self, target, x):
        """ Produces Conditional Expectation:
                E(Y | X1=x1, ... , XN=xN).
            Target(Y) should be a member of the dataset,
            but not a member of the constructor parameter 
            includeVars.
        """
        # Calculate SUM[i=1,N](P(X1[i], ..., XN[i]) * Y[i]) / 
        #   SUM[i=1,N](P(X1[i], ..., XN[i])
        targetVals = self.data[target]
        v1 = 0.0
        v2 = 0.0
        for i in self.nrange:
            tVal = targetVals[i]
            p = self.X[i]
            #print('p = ', p)
            x = np.array(x)
            p = np.array(p)
            density = self.K(p,x)
            v1 += density * tVal
            v2 += density
        # Why 1.07??? Don't know, but it works.
        #return v1 / v2 * 1.07
        if v2 > 0:
            return v1 / v2
        else:
            print('^^^^^^^^^^^^rkhsMV.condE:  V2 = 0', v1, v2)
            return None

    def condCDF(self, x):
        """ 
            Produces Conditional Cumulative Distribution:
                CDF(X1=x1 | X2=x2, ..., XN=xN)

        """
        # d is a small delta value. ds[j] is d * std(Xj)
        # Calculate (CDF(X1, X2+ds2, ..., XN+dsN) - CDF(X1, X2-ds2, ... , XN-dsN)) /
        #       (CDF(X2+ds2, ... , XN+dsN) - CDF(X2-ds2, ... , XN-dsN))
        # P(-inf <= X1 <= x1 | X2=x2, ... , XN=xN) = CDF(X1=x1 | )
        # Is there a simpler formula?
        d = .001
        v1 = 0.0
        v2 = 0.0
        v3 = 0.0
        v4 = 0.0
        condVars = self.varNames[1:]
        if self.R2 is None:
            self.R2 = RKHS(self.data, condVars, s=self.s)
        R2 = self.R2
        mv1 = self.mv
        mv2 = R2.mv
        x2 = x[1:]
        x2offset = d * np.array(R2.stds)
        x2_h = x2 + x2offset # X2+ds2, ... , XN+dsN
        x2_l = x2 - x2offset  # X2-ds2, ... , XN-dsN
        xoffset = np.array([0] + list((d * np.array(self.stds[1:]))))
        x_h = x + xoffset # X1, X2+ds2, XN+dsN
        x_l = x - xoffset # X1, X2-ds2, XN-dsN
        for i in self.nrange:
            p1 = self.X[i]
            p1 = np.array(p1)
            p2 = R2.X[i]
            p2 = np.array(p2)
            c1 = mv1.cdf(x_h - p1) 
            v1 += c1 # CDF(X1, X2+ds2, ..., XN+dsN)
            c3 = mv1.cdf(x_l - p1)
            v3 += c3 # CDF(X1, X2-ds2, ... , XN-dsN)
            c2 = mv2.cdf(x2_h - p2)
            v2 += c2 # CDF(X2+ds2, ..., XN+dsN)
            c4 = mv2.cdf(x2_l - p2)
            v4 += c4 # CDF(X2-ds2, ... , XN-dsN)
        t1 = v1 - v3
        t2 = v2 - v4
        if t2 > .0000001:
            result = min([(t1 / t2), 1])
        else:
            result = 0
        return result