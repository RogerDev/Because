from math import e, sqrt, pi, erf, log
import sys
if '.' not in sys.path:
    sys.path.append('.')
import time
from synth import getData
import matplotlib.pyplot as plt
from RKHSmod.rkhs import RKHS

def Kcdf(self, x1, x2=0):
    #print('x1, x2 = ', x1, x2)
    #result = e**(-((x1-x2)**2) / (2 * self.sigma**2)) / (self.sigma * sqrt(2*pi))
    result = (1.0 + erf((x2-x1) / (self.sigma * sqrt(2.0))))/2
    return result

def cdf(self, x):
    v = 0.0
    for p in self.X:
        v += self.Kcdf(p, x)
        #print('v = ', v)
    return v / len(self.X)


def logistic(x, mu, s):
    return e**(-(x-mu)/s) / (s * ((1 + e**(-(x-mu)/s)))**2)

def logisticCDF(x, mu, s):
    return 1 / (1 + e**(-(x - mu)/s))

def testF(x):
    result = (logistic(x, -2, .5) + logistic(x, 2, .5)  + logistic(x, -.5, .3)) / 3
    return result

def testFCDF(x):
    result = logisticCDF(x, -2, .5) + logisticCDF(x, 2, .5) / 2
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
    sigmas = [1, .5,.3, .2, .1, .05]
    errs = {}

    traces = []
    # Generate a series of evenly spaced test points
    # While we're at it, generate the expected (idealized) result for each test point
    tfs = []
    testPoints = []
    testMin = -5
    testMax = 5
    tp = testMin
    numTP = 1000
    interval = (testMax - testMin) / numTP
    for i in range(numTP + 1):
        testPoints.append(tp)
        tp += interval
        tfp = testF(tp)
        tfs.append(tfp)
        tp += interval
    # Create chart traces of F(x) with various sigmas
    start = time.time()
    delta = 3
    #delta = None
    for sigma in sigmas:
        r = RKHS(X, kparms = [sigma, delta])
        fs = []  # The results of F(x) for each test point
        totalErr = 0
        for i in range(len(testPoints)):
            p = testPoints[i]
            fp = r.F(p)
            fs.append(fp)
            tfp = tfs[i]
            err = abs(fp - tfp)  # F(x) - idealized cdf value
            totalErr += err
        errs[sigma] = totalErr / numTP
        traces.append(fs)
    end = time.time()
    print('elapsed time = ', end - start)
    print('total errors = ', errs)
    #plt.plot(testPoints, ctfs, label='testF(x)', color='#000000', lineWidth=3)
    for t in range(len(traces)):
        fs = traces[t]
        sig = sigmas[t]
        label = 'cdf(X)-sigma=' + str(sig)
        plt.plot(testPoints, fs, label=label)
    plt.plot(testPoints, tfs, label='testF(x)', color='#000000', linewidth=3)
    plt.legend()
    plt.show()

