""" Multivariate RKHS.  
    datapoints is np array of shape (D x N), where D is the number of variables and N is the number of data
    points.
    F(x,M) evaluates Joint Probability Density of all variables except those marked zero in array M.
    Uses Multivariate Gaussian Density Kernel, or Multivariate Gaussian Cumulative Density Kernel, based
    on setting of "cumulative" to True for cumulative or False for density. 
"""
from math import e, sqrt, pi, log 
import numpy as np

class RKHS:
    def __init__(self, data, sigmas = None, includeVars = None, delta = 3, sigmaScale= 1.0, cumulative=False):
        """ kparms is an ordered list of static parameters for the kernel function.
            Different kernels might have a different set of regularization parameters,
            so kparms is a simple ordered list of kernel parameters.
            The default kernel is the Gaussian PDF Kernel, which takes a single
            parameter: sigma, the standard deviation of the Gaussian to use (regularization parameter).
        """
        assert type(data) == type({}), "rkhsMV: Error -- Data must be in the form of a dictionary varName -> [val1, val2, ... , valN]"
        # If includeVars is present, use it to build the samples and associated sigmas
        self.vMap = {} # Map from var name to var index
        self.iMap = {} # Map from index to varName
        if includeVars is None:
            self.varNames = list(data.keys())
            self.D = len(self.varNames)
        else:
            self.varNames = includeVars
            self.D = len(self.varNames)
        self.N = len(data[self.varNames[0]])
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
        # Compute sigma parameter array for kernel
        sigmaList = []
        if sigmas is None:
            # No sigmas provided.  Use our heuristic.
            for i in range(self.D):
                stdDev = self.stdDevs[i]
                sig = 1 / log(self.N, 4) / 3 * stdDev * sigmaScale
                sigmaList.append(sig)
        else:
            for i in range(self.D):
                varName = self.varNames[i]
                sig = sigmas[varName]
                sigmaList.append(sig)
        self.sigmas = np.array(sigmaList)
        self.invSigmas = np.array([1/s for s in self.sigmas])
        self.delta = delta
        self.cumulative = cumulative
        # Calculate the covariance matrix, assuming all vars are independent
        self.covM = np.diag(self.sigmas * self.sigmas)
        # Precompute inverse covariance matrix
        self.covMI = np.linalg.inv(self.covM)
        # Precompute determinant of covM
        self.detcov = np.linalg.det(self.covM)
        # Precompute the denominator for the kernel function
        self.denom = sqrt((2*pi)**self.D * self.detcov) * 2
        print('self.N = ', self.N)
        print('self.D = ', self.D)
        # Precompute range variables for efficency
        self.nrange = range(self.N)
        self.drange = range(self.D)

    
    # Default kernel is gaussian cdf.  Override using k above.
    def K(self, x1, x2=0):
        """ Multivariate gaussian density kernel
            x1 and x2 are vectors of length D.
            returns a density vector of length D.
        """
        diffs = (x1 - x2).reshape(self.D, 1)  # Vector of differences
        #diffs = np.array([0] * self.D).reshape(self.D, 1)
        return e**(-.5 * (diffs.transpose() @ self.covMI @ diffs)[0][0]) / self.denom
        #return density

    def K_temp(self, x1, x2=0):
        """ Multivariate gaussian density kernel
            x1 and x2 are vectors of length D.
            returns a density vector of length D.
        """
        diffs = []
        for d in range(self.D):
            diffs.append(x1[d] - x2[d])
        mahal = 0
        if self.D == 1:
            v1 = diffs[0]
            mahal = v1**2 / self.covM[0,0]
        else:
            v1 = diffs[0]
            v2 = diffs[1]
            s21 = self.covM[0,0]
            s22 = self.covM[1,1]
            mahal += v1**2/s21
            mahal += v2**2/s22
            mahal += v1 * v2 / s21
            mahal += v1 * v2 / s22
        density = e**(-.5 * mahal) / self.denom
        return density

    def F(self, x):
        """ x is a multivariate sample of length D
        """
        delta = self.delta
        v = 0.0
        N = self.N
        if not self.cumulative:
            for p in self.X:
                dist2 = 0
                for j in self.drange:
                    dist2 += ((p[j]-x[j]) * self.invSigmas[j])**2
                if self.delta is not None:
                    dist = sqrt(dist2)
                    if dist > delta:
                        continue
                v += e**(-.5 * dist2) / self.denom
            result = v / N
            return result


