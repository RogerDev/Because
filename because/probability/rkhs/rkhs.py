from math import e, sqrt, pi 
import numpy as np

class RKHS:
    def __init__(self, dataPoints, k=None, f=None, kparms=[]):
        """ kparms is an ordered list of static parameters for the kernel function.
            Different kernels might have a different set of regularization parameters,
            so kparms is a simple ordered list of kernel parameters.
            The default kernel is the Gaussian PDF Kernel, which takes a single
            parameter: sigma, the standard deviation of the Gaussian to use (regularization parameter).
        """
        if type(dataPoints) == type([]) and 0:
            # Input data is in list form.  Convert to np.array.
            self.X = np.array(dataPoints)
        else:
            # Already an np.array
            self.X = dataPoints

        # k overrides the kernel function K
        # k should take three parameters (x1, x2, kparms)
        self.k = k
        # f overrides the evaluation function F(x)
        # f should have 3 positional parameters (x, data, kparms)
        self.f = f
        self.kparms = kparms
    
    # Default kernel is gaussian cdf.  Override using k above.
    def K(self, x1, x2=0):
        if self.k is None:
            # No kernel provided.  Use the Gaussian Kernel
            diff = x1 - x2
            sigma = self.kparms[0]
            result = e**(-((diff)**2) / (2 * sigma**2)) / (sigma * sqrt(2*pi))
        else:
            # Call the user-provided kernel
            result = self.k(x1, x2, self.kparms)
        return result
    
    def F(self, x):
        if self.f is None:
            # Use the standard evaluation method
            v = 0.0
            sigma = self.kparms[0]
            if len(self.kparms) > 1:
                delta = self.kparms[1]
            else:
                delta = None
            for p in self.X:
                if delta is not None and abs(p - x) > delta * sigma :
                    continue
                v += self.K(p, x)
            return v / len(self.X)
        else:
            # Use the user-provided method
            return self.f(x, self.X, self.kparms)

