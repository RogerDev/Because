from math import e, sqrt, pi, ceil, log, erf
import numpy as np
from scipy.special import erfinv

class RKHS:
    def __init__(self, dataPoints, rff_dim=None, s=None):
        """ kparms is an ordered list of static parameters for the kernel function.
            Different kernels might have a different set of regularization parameters,
            so kparms is a simple ordered list of kernel parameters.
            The default kernel is the Gaussian PDF Kernel, which takes a single
            parameter: sigma, the standard deviation of the Gaussian to use (regularization parameter).
        """
        # self.X should be an N X 1 array
        if type(dataPoints) == type([]):
            # Input data is in list form.  Convert to np.array.
            self.X = np.array(dataPoints).reshape((len(dataPoints), 1))
        else:
            # Already an np.array
            self.X = dataPoints.reshape((dataPoints.shape[0], 1))
        self.N = self.X.shape[0]
        if s is None:
            s = 1.0
        scale = (.5 * s * self.N**(-1/6))

        if rff_dim is None:
            rff_dim = max(ceil(self.N / 40), 100)
            #rff_dim = max(ceil(sqrt(self.N)*3), 100)
            #rff_dim = 50
            #rff_dim = ceil(sqrt(self.N))
            #rff_dim = ceil(self.N /.1)
            #rff_dim = 1000
        self.rff_dim = rff_dim
        #self.rff_dim = 1000
        self.W_ = None
        self.b_ = None

        self.sigma2 = np.std(self.X)
        self.sigma = scale
        #print('sigma = ', self.sigma)
        self.delta = 3
        #print('sigma = ', self.sigma)
        self.Z_ = None

    # Default kernel is gaussian cdf.  Override using k above.
    def Kpdf(self, x1, x2=0):
        # PDF Kernel
        sigma = self.sigma
        diff = x1 - x2
        result = e**(-((diff)**2) / (2 * sigma**2)) / (sigma * sqrt(2*pi))
        return result

    def Kcdf(self, x1, x2=0):
        # CDF Kernel
        return (1 + erf(abs(x2-x1)/self.sigma*sqrt(2))) / 2

    # kType is 'p' for PDF or 'c' for CDF, 's' for standard deviation,  or 'e' for Expectation.  Default: 'p'
    def F(self, x, kType='p'):
        v = 0.0
        sigma = self.sigma
        delta = self.delta
        dsigma = sigma * delta
        assert kType == 'p' or kType == 'c' or kType == 's' or kType == 'e', "kType must be 'c' or 'p', 's' or 'e'.  Received: " + str(kType) + '.'
        if kType == 'p':
            for p in self.X:
                if delta is not None and abs(p - x) > dsigma :
                    continue
                v += self.Kpdf(p, x)
        elif kType == 'c':
            for p in self.X:
                if p < x:
                    if delta is not None and (x-p) > dsigma:
                        v += 1
                        continue
                    v += self.Kcdf(p, x)
                else:
                    if delta is not None and (p-x) > dsigma:
                        continue
                    v += 1 - self.Kcdf(p, x)
        elif kType == 's':
            # Standard Deviation
            cumE = 0.0   # Sum of means
            cumE2 = 0.0  # Sum of squared means
            for p in self.X:
                cumE += p
                cumE2 += p**2
            var = self.N * self.sigma + cumE2 - cumE**2
            std = sqrt(var)
            return std

        elif kType == 'e':
            # Expectation
            for p in self.X:
                v += p
        return float(v) / len(self.X)


    def Frff(self, x, kType='p'):
        assert kType == 'p' or kType == 'c', "kType must be 'c' or 'p'.  Received: " + str(kType) + '.'
        if kType == 'p':
            if type(x) != type([]):
                x = [x]
            X = np.array([x]).reshape(len(x),1)
            Z = self._get_rffs(X)
            #print('Z.shape = ', Z.shape)
            K = (Z.T @ Z)
            #(2 * sigma**2)) / (sigma * sqrt(2*pi)
            vals = np.sum( K, 1)
            #v = np.sum(vals) / self.N
            #v = np.sum(vals) / self.N
            #return float(v) * log(self.rff_dim)
            vals = list(vals)
            vals2 = [max([0, val])/self.N * pi/2 for val in vals]
            return vals2[0]
        else:
            # No RFF for CDF yet.  Just use F.
            return self.F(x, kType)

    def Frff_old(self, x, kType='p'):
        assert kType == 'p' or kType == 'c', "kType must be 'c' or 'p'.  Received: " + str(kType) + '.'
        if kType == 'p':
            if self.Z_ is None:
                Z = self._get_rffs(self.X, return_vars=False)
                self.Z_ = Z
                print('Z, W, b = ', self.Z_.shape, self.W_.shape, self.b_.shape)
            if type(x) != type([]):
                x = [x]
            X = np.array([x]).reshape(len(x),1)
            Z = self._get_rffs(X)
            #print('Z.shape = ', Z.shape)
            K = (Z.T @ self.Z_)
            #(2 * sigma**2)) / (sigma * sqrt(2*pi)
            vals = np.sum( K, 1)
            #v = np.sum(vals) / self.N
            #v = np.sum(vals) / self.N
            #return float(v) * log(self.rff_dim)
            vals = list(vals)
            vals2 = [max([0, val])/self.N * pi/2 for val in vals]
            return vals2
        else:
            # No RFF for CDF yet.  Just use F.
            return self.F(x, kType)

    def _get_rffs(self, X, return_vars=False):
        """Return random Fourier features based on data X, as well as random
        variables W and b.
        """
        N, D = X.shape
        #print(X.shape)
        if self.W_ is not None:
            W, b = self.W_, self.b_
        else:
            np.random.seed(5555)
            W = np.random.normal(loc=0, scale=1, size=(self.rff_dim, D))
            b = np.random.uniform(0, 2*np.pi, size=self.rff_dim)
            self.W_ = W
            self.b_ = b

        B = np.repeat(b[:, np.newaxis], N, axis=1)
        norm = 1./ np.sqrt(self.rff_dim)
        #norm = 1./self.rff_dim
        #(2 * sigma**2)) / (sigma * sqrt(2*pi)
        sigma = 1 / self.sigma
        #sigmaF = (2 * sigma**2) / (sigma * sqrt(2*pi))
        sigmaF = sigma / self.Kpdf(self.sigma)
        #sigmaF = sigma
        print('Ksig = ', self.Kpdf(self.sigma))
        #print('sigmaF = ', sigmaF)
        #Z    = norm * np.sqrt(2) * np.cos((sigma * W) @ X.T + B)# * (2*self.sigma**2) / sqrt(2*pi)
        Z    = norm * np.sqrt(2) * np.cos((sigmaF * W) @ X.T + B) # Z is rff_dim X N
        if return_vars:
            return Z, W, b
        return Z
