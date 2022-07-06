"""
Plot a 3D graph of a single conditional probability
i.e., P(A2 | B), using the probability/test/models/nCondition.py SEM file.
Automatically generates data for each run.
Plots two graphs, comparing different variations of ProbSpace
cMethod parameter.  See ProbSpace for details on the different
methods.
Set the method using prob1 and prob2 constructors below.
"""
import sys
if '.' not in sys.path:
    sys.path.append('.')
import time
from math import log, tanh, sqrt, sin, cos, e

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from because.causality import rv
from because.synth import read_data, gen_data
from because.probability import independence
from because.probability.prob import ProbSpace
from because.probability.rkhs.rkhsMV import RKHS
from because.visualization import grid

def show(dataPath='', numRecs=0, probspace=None):
    # Arg format is  <datSize>
    dims = 1
    if probspace is None:        

        f = open(dataPath, 'r')
        exec(f.read(), globals())

        print('Testing: ', dataPath, '--', testDescript)

        # Generate the data
        gen = gen_data.Gen(dataPath)
        data = gen.getDataset(numRecs)

        prob1 = ProbSpace(data, cMethod = 'd!')
    else:
        prob1 = probspace

    numPts = 100
    lim = .1  # Percentile at which to start.  Will end at percentile(100-lim)

    print('Test Limit = ', lim, 'percentile to' , 100-lim, 'percentile.')
    N = prob1.N
    print('N = ', N)
    vars = prob1.fieldList[:5]
    yTraces = []
    minmin = 9999999
    maxmax = -9999999
    for var in vars:
        print('var = ', var)
        d = prob1.distr((var,))
        minv = d.minVal()
        maxv = d.maxVal()
        if minv < minmin:
            minmin = minv
        if maxv > maxmax:
            maxmax = maxv
    tps = np.linspace(minmin, maxmax, numPts)
    dp_start = time.time()
    xTrace = []
    for i in range(len(tps) -1):
        t = tps[i]
        xTrace.append(t)
    for v in vars:
        yTrace = []
        for i in range(len(tps)-1):
            t = tps[i]
            nextT = tps[i+1]
            py = prob1.P((v, t, nextT))
            if py is None or py < .001 / numPts:
                py = 0
            yTrace.append(py)
        yTraces.append(yTrace)
    dp_end = time.time()
    results = []
    ysum = 0.0

    fig = plt.figure(tight_layout=True)


    my_cmap = plt.get_cmap('tab20')
    ax = fig.add_subplot(111)
    v1Label = '$P(Var=X)$'
    v2Label = '$' + 'X' + '$'
    colors = my_cmap.colors
    ax.set_ylabel(v1Label, fontweight='bold', rotation = 90)
    ax.set_xlabel(v2Label, fontweight='bold', rotation = 0)
    x = np.array(xTrace)
    print('x = ', x.shape)
    for i in range(len(yTraces)):
        y = np.array(yTraces[i])
        print('y = ', y.shape)
        ax.fill_between(x, y, 0, color=colors[i], alpha=.3)
        ax.plot(x, y, color=colors[i], linewidth=2, alpha=.7)
    plt.show()

if __name__ == '__main__':
    if '-h' in sys.argv or len(sys.argv) < 2:
        print('Usage: python because/visualization/probPlot1D.py dataPath varName [numRecs] [cum]')
        print('  dataPath is the path to a .py (synthetic data) or .csv file')
    else:
        numRecs = 0 
        args = sys.argv
        dataPath = args[1].strip()
        if len(args) > 2:
            try:
                numRecs = int(args[2].strip())
            except:
                pass
        #print('dims, datSize, numRecs = ', dims, datSize, numRecs)
        show(dataPath=dataPath, numRecs=numRecs)
