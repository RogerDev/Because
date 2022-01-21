"""
    Test to generate and plot the density of a nonparametric sample, along side a resampling of the sample.
    Process:
        Use RKHS to generate PDF function of sample (pdf(S))
        Use RKHS to generate the CDF function of the sample (cdf(S)) 
        Sample n points (S2) from the cdf (p = cdf(x)), using a uniform distribution and create 
            inverse mapping function: icdf: p -> x. This is equivalent to the
            quantile function of the nonparametric distribution.  We use this
            discretized map because we could not find an RKHS kernel that would
            replicate the quantile function.
        Use RKHS to generate the PDF function (dS2) of the resampled distribution (S2)
            (i.e. PDF(icdf(uniform(0,1,n))))
        Graph dS vs dS2.
"""
from math import e, sqrt, pi, erf, erfc, log
from scipy.special import erfinv
import sys
if '.' not in sys.path:
    sys.path.append('.')
import time
from synth import getData
import matplotlib.pyplot as plt
from RKHSmod.rkhs import RKHS
from numpy import random
import numpy as np
import math

# Note: Could not make quantile kernel work.  Not square integrable.
def kQuant(x1, x2=0, kparms=[]):
    sigma = 1
    gamma = .5
    if kparms:
        sigma = kparms[0]
    return sigma * sqrt(2) * erfinv(2*(abs(x1-x2)/2 + .5) - 1)

# Symmetric Kernel of cdf function = normal cdf.
def kcdf(x1, x2=0, kparms=[]):
    # CDF Kernel
    sigma = kparms[0]
    
    return (1 + erf(abs(x2-x1)/sigma*sqrt(2))) / 2

# Compute the RKHS Mean of the CDF.  There is a trick here.
# Since kernel is required to be symmetric, we negate the result
# whenever x < Xi.  Standard trick for creating asymetric results from
# a symmetric kernel
def Fcdf(x,X, kparms):
    sigma = kparms[0]
    delta = kparms[1]
    cum = 0.0
    for p in X:
        if p < x:
            if delta is not None and (x-p) > delta * sigma:
                cum += 1
                continue
            cum += kcdf(p, x, kparms)
        else:
            if delta is not None and (p-x) > delta * sigma:
                continue
            cum += 1 - kcdf(p, x, kparms)
    return cum / len(X)

# RKHS Mean function for quant.  Uses same trick as for CDF above, but it
# doesn't quite work.
def Fquant(x,X, kparms):
    sigma = kparms[0]
    delta = kparms[1]
    cum = 0.0
    gamma =1
    cnt = 0
    for p in X:
        if abs(p-x) > gamma:
            continue
        v = kQuant(p, x, kparms)
        if x > p:
            cum += v
        else:
            cum += -v
        cnt+=1
    return cum / cnt

def logistic(x, mu, s):
    return e**(-(x-mu)/s) / (s * ((1 + e**(-(x-mu)/s)))**2)

def logisticCDF(x, mu, s):
    return 1 / (1 + e**(-(x - mu)/s))

def testF(x):
    result = (logistic(x, -2, .5) + logistic(x, 2, .5)  + logistic(x, -.5, .3)) / 3
    return result

def testFCDF(x):
    result = (logisticCDF(x, -2, .5) + logisticCDF(x, 2, .5) + logisticCDF(x, -.5, .3)) / 3
    return result

if __name__ == '__main__':
    args = sys.argv
    if (len(args) > 1):
        test = args[1]
    else:
        test = 'models/rkhsTest.csv'
    d = getData.DataReader(test)
    data = d.read()
    X = data['X'][:10000]
    tps = [] # Uniform test points
    testMin = -5
    testMax = 5
    tp = testMin
    numTP = 200 # Number of test points for graphing
    interval = (testMax - testMin) / numTP
    X2 = [] # Resampled X
    stdx = np.std(X) # Std Dev of X
    # Generate a uniform range of test points, from testMin to testMax.
    for i in range(numTP + 1):
        tps.append(tp)
        tp += interval
    sigma = 1 / log(len(X), 4)  # Heuristic value for Sigma in the Gaussian kernel.
    delta = 3 # Ignore points more than delta from the mean.  Optimization.
    # RKHS for the cdf function.  rcdf.F(x) = cdf(x)
    rcdf = RKHS(X, f = Fcdf, k=kcdf, kparms = [sigma, delta])
    # Uniformly sample the cdf to generate a series of p -> x mappings
    U = np.random.uniform(-5, 5, 1000)
    p2x = []
    for i in range(len(U)):
        x = U[i]
        p = rcdf.F(x)
        p2x.append((p, x))
    p2x.sort()
    # Function to retrieve the interpolated mapping from p2x.  icdf(p) = quantile(p)
    def icdf(p):
        x = p2x[-1][1]
        prevT = p2x[0]
        for j in range(len(p2x)):
            t = p2x[j]
            pj,xj = t
            if pj >= p:
                prevP, prevX = prevT
                if (pj-prevP) > 0:
                    x = (prevX * (p-prevP) + xj * (pj - p)) / (pj - prevP)
                else:
                    x = xj
                break
            prevT = t
        return x
    # Prepare 3 traces for the graph
    fp1 = []  # PDF(S)
    fp2 = []  # PDF(S2)
    fp3 = []  # CDF(S)
    # Generate new sample X2 = S2 by uniformly
    # sampling the quantile (icdf) function.  Generate
    # n samples.
    for i in range(len(X)):
        t = random.uniform(0,1)
        samp = icdf(t)
        X2.append(samp)
    start = time.time()
    # RKHS for pdf(S)
    r1 = RKHS(X, kparms=[sigma, delta])
    # RKHS for pdf(S2)
    r2 = RKHS(X2, kparms=[sigma, delta])
    # Prepare 3 traces for the graph
    fp1 = []  # PDF(S)
    fp2 = []  # PDF(S2)
    fp3 = []  # CDF(S)
    for i in range(len(tps)):
        x = tps[i]
        fp1.append(r1.F(x))
        fp3.append(rcdf.F(x))
        fp2.append(r2.F(x))
    end = time.time()
    print('elapsed = ', end - start)
    plt.plot(tps, fp1, label='Original', linestyle='solid')
    #plt.plot(tps, fp3, label='Original CDF', linestyle='solid')
    plt.plot(tps, fp2, label='Resampled', linestyle='dashed', color='red')
    plt.legend()
    plt.show()

