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
from math import log, tanh, sqrt, sin, cos, e, ceil

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
    # Arg format is  <datSize>
    dims = 1
    if probspace is None:
        # Got a .csv or .py file
        tokens = dataPath.split('.')
        ds = None # The dataset in dictionary form
        assert len(tokens) == 2 and (tokens[1] == 'py' or tokens[1] == 'csv'), 'Viz.show: dataPath must have a .py or .csv extension.  Got: ' + dataPath
        if tokens[1] == 'py':
            # py SEM file
            assert numRecs > 0, 'Viz.show: For synthetic data (i.e. .py extension) numRecs must be positive'
            gen = gen_data.Gen(dataPath)
            ds = gen.getDataset(numRecs)
        else:
            # csv
            r = read_data.Reader(dataPath)
            ds = r.read()
        probspace = ProbSpace(ds)
    prob1 = probspace
    numPts = 20
    lim = .1  # Percentile at which to start.  Will end at percentile(100-lim)

    #print('Test Limit = ', lim, 'percentile to' , 100-lim, 'percentile.')
    vars = [spec[0] for spec in prob1.normalizeSpecs(targetSpec)]
    N = prob1.N
    #print('N = ', N)
    if not vars:
        vars = prob1.fieldList
    print('vars = ', vars)
    means = []
    pls = []
    plls = []
    phs = []
    phhs = []
    for var in vars:
        if not prob1.isCategorical(var):
            dist = prob1.distr(var)
            means.append(dist.E())
            pls.append(dist.percentile(2.5))
            plls.append(dist.percentile(16))
            phs.append(dist.percentile(97.5))
            phhs.append(dist.percentile(84))
        else:
            means.append(0)
            pls.append(0)
            plls.append(0)
            phs.append(0)
            phhs.append(0)
    xTraces = []
    yTraces = []
    for var in vars:
        yTraces.append([])
        xTraces.append([])
    start = time.time()
    for i in range(len(vars)):
        var = vars[i]
        g = grid.Grid(prob1, [var], lim, numPts)
        tps = g.makeGrid()
        incrs = g.getIncrs()
        incr = incrs[0]
        isCat = prob1.isCategorical(var)
        isStr = prob1.isStringVal(var)
        for tp in tps:
            t = tp[0]
            if isCat:
                if isStr:
                    t = prob1.numToStr(var, t)
                tSpec = (var, t)
            else:
                tSpec = (var, t, t+incr)
            p = prob1.P(tSpec)
            xTraces[i].append(t)
            yTraces[i].append(p)
    end = time.time()
    cols = 2
    rows = ceil(len(vars) / cols)
 
    
    # Initialize the figure
    #plt.style.use('seaborn-darkgrid')
    pltcolors = plt.get_cmap('tab10').colors
    colors = plt.get_cmap('tab20').colors
    green = (0, .6, 0)
    yellow = colors[17]
    # create a color palette
    fig, axs = plt.subplots(rows,cols, tight_layout=True)
    axes = axs.flat
    def sortByProb(x,y):
        xy = [(y[i],x[i]) for i in range(len(x))]
        xy.sort()
        xy.reverse()
        x = [xy[i][1] for i in range(len(x))]
        y = [xy[i][0] for i in range(len(x))]
        return x, y
    for i in range(len(vars)):
        var = vars[i]
        ax = axes[i]
        x = np.array(xTraces[i])
        y = np.array(yTraces[i])
        if prob1.isCategorical(var):
            x, y = sortByProb(x, y)
        linecolor = pltcolors[i % len(pltcolors)]
        ax.fill_between(x, y, 0, color=linecolor, alpha=.7)
        ax.plot(x, y, color=linecolor, linewidth=2, alpha=1)
        if not prob1.isCategorical(var):
            mean = means[i]
            ph = phs[i]
            phh = phhs[i]
            pl = pls[i]
            pll = plls[i]
            ax.axvspan(phh, pll, color=linecolor, alpha=.1)
            ax.axvspan(ph, pl, color=linecolor, alpha=.1)
            ax.axvline(x=ph, color=yellow, alpha=1, linewidth = 1)
            ax.axvline(x=pl, color=yellow, alpha=1, linewidth = 1)
            ax.axvline(x=phh, color=yellow, alpha=1, linewidth = .75)
            ax.axvline(x=pll, color=yellow, alpha=1, linewidth = .75)
            ax.axvline(x=mean, color=green, linewidth = 2)
            ax.axvspan(phh, pll, color=linecolor, alpha=.2)
            ax.axvspan(ph, pl, color=linecolor, alpha=.2)
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(0, np.max(y) + .01)
        ax.set_title(var, loc='left', fontsize=12, fontweight=0, color='black')
        ax.tick_params(axis='x', rotation = -60)
    #plt.subplots_adjust(hspace=.5)
    plt.show()

if __name__ == '__main__':
    if '-h' in sys.argv or len(sys.argv) < 3:
        print('Usage: python because/visualization/probPlotAll.py dataPath varNames [numRecs] [cum]')
        print('  dataPath is the path to a .py (synthetic data) or .csv file')
    else:
        numRecs = 0 
        args = sys.argv
        dataPath = args[1].strip()
        varStr = args[2].strip()
        if varStr:
            tokens = varStr.split(',')
            vars = [tok.strip() for tok in tokens]
        else:
            vars = []
        if len(args) > 3:
            try:
                numRecs = int(args[3].strip())
            except:
                pass
        #print('dims, datSize, numRecs = ', dims, datSize, numRecs)
        show(dataPath=dataPath, targetSpec=vars, numRecs=numRecs)
