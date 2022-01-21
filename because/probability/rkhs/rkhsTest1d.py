from math import e, sqrt, pi, erf, erfc, log
from scipy.special import erfinv
import sys
if '.' not in sys.path:
    sys.path.append('.')
import time
from synth import getData
import matplotlib.pyplot as plt
from RKHSmod.rkhsRFF import RKHS
from Probability import Prob

def kQuant(x1, x2=0, kparms=[]):
    sigma = kparms[0]
    return sigma * sqrt(2) * erfinv(2*abs(x1-x2)-1)

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
    kType = 'c'
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
    dataSizes = [5, 25, 125, 625, 3125]
    maxErrR = {}
    maxErrD = {}
    avgErrR = {}
    avgErrD = {}
    testPoints = []
    testMin = -10
    testMax = 10
    tp = testMin
    numTP = 200
    interval = (testMax - testMin) / numTP
    tfs = []
    ctfs = []
    means = {}
    # Generate a uniform range of test points.
    # While at it, generate our expected pdf and cdf
    for i in range(numTP + 1):
        testPoints.append(tp)
        tfp = testF(tp)
        tfs.append(tfp)
        ctfp = testFCDF(tp)
        ctfs.append(ctfp)
        tp += interval
    
    evals = 0
    for size in dataSizes:
        elapsedR = 0.0
        elapsedD = 0.0
        r1 = RKHS(X[:size], rff_dim=size)
        data = {'X':X[:size]}
        startD = time.time()
        ps = Prob.ProbSpace(data)
        distr = ps.distr('X')
        endD = time.time()
        eD = endD - startD
        elapsedD += eD        
        #r3 = RKHS(X[:size], f=Fquant, k=kQuant, kparms=[sigma, None])
        ds = []  # The results using DPROB
        fs = []  # The results using rkhs
        errsR = []
        errsD = []

        deviations = []
        for i in range(len(testPoints)):
            p = testPoints[i]
            startR = time.time()
            if kType == 'p':
                fp = r1.F(p, kType)
            else:
                fp = r1.F(p, kType)
            endR = time.time()
            eR = endR - startR
            elapsedR += eR
            evals += 1
            fs.append(fp)
            startD = time.time()
            if kType == 'p':
                tfp = tfs[i]
                dsp = distr.P(p)
            else:
                tfp = ctfs[i]
                dsp = distr.P((-999999, p))
            endD = time.time()
            eD = endD - startD
            elapsedD += eD
            ds.append(dsp)
            errR = abs(fp - tfp)
            errD = abs(dsp - tfp)
            errsR.append(errR)
            errsD.append(errD)
        maxErrR[size] = max(errsR)
        avgErrR[size] = sum(errsR) / numTP
        maxErrD[size] = max(errsD)
        avgErrD[size] = sum(errsD) / numTP
        traces.append(fs) # rkhs trace
        traces.append(ds) # D-Prob trace
        #mean = r3.F(.5)
        #means[size] = mean
    print('Average ErrorsR = ', avgErrR)
    print('Average ErrorsD = ', avgErrD)
    print('Maximum ErrorsR = ', maxErrR)
    print('Maximum ErrorsD = ', maxErrD)
    print('elapsedR = ', elapsedR)
    print('elapsedD = ', elapsedD)
    print('perEvalR = ', elapsedR / evals)
    print('perEvalD = ', elapsedD / evals)

    colors = ['red', 'orange', 'brown', 'blue', 'green']
    for t in range(len(traces)):
        fs = traces[t]
        size = dataSizes[int(t/2)] # traces are alternately rkhs and d-prob
        cindex = int(t/2)
        color = colors[cindex]
        if t%2 == 0:
            typ = 'RKHS'
            linestyle = 'solid'
        else:
            typ = 'DPROB'
            linestyle = 'dashed'
        if kType == 'p':
            label = typ + '-pdf(X)-size=' + str(size)
        else:
            label = typ + '-cdf(X)-size=' + str(size)
        plt.plot(testPoints, fs, label=label, linestyle=linestyle, color=color)
    if kType == 'p':
        plt.plot(testPoints, tfs, label='testPDF(x)', color='#000000', linewidth=3, linestyle='dashed')
    else:
        plt.plot(testPoints, ctfs, label='testCDF(x)', color='#000000', linewidth=3, linestyle='dashed')
    plt.legend()
    plt.show()

