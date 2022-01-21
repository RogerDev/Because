from math import e, sqrt, pi, erf, erfc, log
from scipy.special import erfinv
import sys
if '.' not in sys.path:
    sys.path.append('.')
import time
from synth import getData
import matplotlib.pyplot as plt
from RKHSmod.rkhsRFF import RKHS

def kQuant(x1, x2=0, kparms=[]):
    sigma = kparms[0]
    return sigma * sqrt(2) * erfinv(2*abs(x1-x2)-1)

def kcdf(x1, x2=0, kparms=[]):
    # CDF Kernel
    sigma = kparms[0]
    return (1 + erf(abs(x2-x1)/sigma*sqrt(2))) / 2

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

def Fquant(x,X, kparms):
    sigma = kparms[0]
    delta = kparms[1]
    cum = 0.0
    for p in X:
        if x > p:
            if delta is not None and (x-p) > delta * sigma:
                cum += 1
                continue
            cum += kQuant(p, x, kparms)
        else:
            if delta is not None and (p-x) > delta * sigma:
                continue
            cum += -kQuant(p, x, kparms)
    return cum / len(X)

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
    X = data['X']
    expmean = sum(X) / len(X)
    traces = []
    #dataSizes = [10, 50, 100, 1000, 10000, 100000]
    dataSizes = [100, 1000, 10000]
    errs = {}
    errsR = {}
    testPoints = []
    testMin = -5
    testMax = 5
    tp = testMin
    numTP = 200
    interval = (testMax - testMin) / numTP
    tfs = []
    ctfs = []
    means = {}
    # Generate a uniform range of test points.
    # While at it, generate our expected pdf
    for i in range(numTP + 1):
        testPoints.append(tp)
        tfp = testF(tp)
        tfs.append(tfp)
        tp += interval
    delta = 2
    start = time.time()
    evals = 0
    for size in dataSizes:
        # Choose a reasonable sigma based on data size.
        r1 = RKHS(X[:size])
        fs = []  # The results using a pdf kernel
        totalErr = 0
        for i in range(len(testPoints)):
            p = testPoints[i]
            tfp = tfs[i] # The ground truth
            fp = r1.F(p)
            evals += 1
            fs.append(fp)
            err = abs(fp - tfp)
            totalErr += err
            #print('fpc, ctfp = ', fpc, ctfp)
        errs[size] = totalErr / numTP
        traces.append(fs) # rkhs trace
    end = time.time()
    # End of RKHS testing
    # Start RFF testing
    startR = time.time()
    for size in dataSizes:
        # Choose a reasonable sigma based on data size.
        r1 = RKHS(X[:size])
        fsR = [] # The result using RFFs
        totalErrR = 0
        fsR = r1.Frff(testPoints)
        #print('fsR = ', len(fsR), fsR)
        for i in range(len(testPoints)):
            
            tfp = tfs[i] # The ground truth
            #fpR = r1.Frff(p)
            #print('fpR = ', fpR)
            fpR = fsR[i]
            errR = abs(fpR - tfp)
            totalErrR += errR
            #print('fpc, ctfp = ', fpc, ctfp)
        errsR[size] = totalErrR / numTP

        traces.append(fsR) # RFF trace
        #mean = r3.F(.5)
        #means[size] = mean
    endR = time.time()
    print('Average Errors (RKHS) = ', errs)
    print('Average Errors (RFF) = ', errsR)

    #print('Means = ', means, expmean)

    duration = end - start
    durationR = endR - startR
    print('elapsed (rkhs) = ', duration)
    print('elapsed (RFF) = ', durationR)
    print('evaluations = ', evals)
    print('elapsed per eval = ', duration / evals)
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    numEach = int(len(traces)/2)
    for t in range(len(traces)):
        fs = traces[t]
        color = colors[t%numEach]
        size = dataSizes[int(t%len(dataSizes))]
        if t<numEach:
            label = 'rkhs.pdf(X), size=' + str(size)
            linestyle = 'solid'
        else:
            label = 'rff.pdf(X), size=' + str(size)
            linestyle = 'dashed'
        plt.plot(testPoints, fs, label=label, linestyle=linestyle, color=color)
    plt.plot(testPoints, tfs, label='testPDF(x)', color='#000000', linewidth=3, linestyle='solid')
    #plt.plot(testPoints, ctfs, label='testCDF(x)', color='#000000', linewidth=3, linestyle='dotted')
    plt.legend()
    plt.show()

