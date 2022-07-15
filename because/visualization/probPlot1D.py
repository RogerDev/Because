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

def show(dataPath='', numRecs=0, targetSpec=[], condSpec=[], gtype='pdf', probspace=None):
    assert len(condSpec) == 0 and len(targetSpec) == 1, 'ProbPlot1D.show: a single target variable is required and no conditions are supported.'
    tries = 1
    datSize = 200
    # Arg format is  <datSize>
    dims = 1
    smoothness = 1
    cumulative = False
    if gtype == 'cdf':
        cumulative = True
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

    v1 = targetSpec[0][0]
    numPts = 30
    lim = .1  # Percentile at which to start.  Will end at percentile(100-lim)

    print('Test Limit = ', lim, 'percentile to' , 100-lim, 'percentile.')
    N = prob1.N
    # P(A2 | B)
    target = v1
    dist = prob1.distr(v1)
    mean = dist.E()
    std = dist.stDev()
    skew = dist.skew()
    kurt = dist.kurtosis()
    pl = dist.percentile(2.5)
    pll = dist.percentile(16)
    ph = dist.percentile(97.5)
    phh = dist.percentile(84)
    # Get test points
    g = grid.Grid(prob1, [v1], lim, numPts)
    tps = g.makeGrid()
    incrs = g.getIncrs()
    incr = incrs[0]
    xt1 = []
    xt2 = []
    yt1 = []
    yt2 = []

    target = v1
    dp_start = time.time()
    for tp in tps:
        t = tp[0]
        if cumulative:
            d = prob1.distr(target)
            if d is None:
                py = 0
            else:
                py = d.P((None, t))
        else:
            if prob1.isDiscrete(target):
                tSpec = (target, t)
            else:
                tSpec = (target, t, t+incr)

            py = prob1.P(tSpec)
            if py is None:
                continue
            if py > 1:
                print('got p > 1:  t = ', t)
                py = 1
        xt1.append(t)
        yt1.append(py)
    dp_end = time.time()
    results = []
    ysum = 0.0

    fig = plt.figure(tight_layout=True)


    x = np.array(xt1)
    y = np.array(yt1)

    my_cmap = plt.get_cmap('tab20')
    ax = fig.add_subplot(111)
    if cumulative:
        v1Label = '$P( 0 <= '  + v1 + ' < X )$'
    else:
        v1Label = '$P(' + v1 + '=X)$'
    v2Label = '$' + 'X' + '$'
    colors = my_cmap.colors
    blue = colors[0]
    gray = colors[15]
    dkgray = colors[14]
    green = colors[5]
    yellow = colors[17]
    ax.set(title = "Mean = " + str(round(mean, 3)) + ', std = ' + str(round(std, 3)) + ', skew = ' + str(round(skew, 3)) + ', kurt = ' + str(round(kurt, 3)))
    ax.set_ylabel(v1Label, fontweight='bold', rotation = 90)
    ax.set_xlabel(v2Label, fontweight='bold', rotation = 0)
    ax.axvspan(phh, pll, color=gray, alpha=.1)
    ax.axvspan(ph, pl, color=gray, alpha=.1)
    plt.axvline(x=ph, color=yellow, alpha=1, linewidth = 1)
    plt.axvline(x=pl, color=yellow, alpha=1, linewidth = 1)
    plt.axvline(x=phh, color=yellow, alpha=1, linewidth = .75)
    plt.axvline(x=pll, color=yellow, alpha=1, linewidth = .75)
    plt.axvline(x=mean, color=green, linewidth = 2)
    ax.axvspan(phh, pll, color=gray, alpha=.2)
    ax.axvspan(ph, pl, color=gray, alpha=.2)
    ax.fill_between(x, y, 0, color=blue, alpha=.3)
    ax.plot(x, y, color=blue, linewidth=2)
    plt.show()

if __name__ == '__main__':
    if '-h' in sys.argv or len(sys.argv) < 3:
        print('Usage: python because/visualization/probPlot1D.py dataPath varName [numRecs] [cum]')
        print('  dataPath is the path to a .py (synthetic data) or .csv file')
        print('  varName is the variable whose distribution to plot.')
        print('  numRecs is the number of records to generate')
        print('  "cum" as a final parameter causes display of cumulative distributions.  Otherwise PDFs are shown')
    else:
        numRecs = 0 
        args = sys.argv
        dataPath = args[1].strip()
        varName = args[2].strip()
        if len(args) > 3:
            try:
                numRecs = int(args[3].strip())
            except:
                pass
        if 'cum' in args:
            gtype = 'cdf'
        else:
            gtype = 'pdf'
        #print('dims, datSize, numRecs = ', dims, datSize, numRecs)
        show(dataPath=dataPath, numRecs=numRecs, targetSpec=[(varName,)], gtype=gtype)
