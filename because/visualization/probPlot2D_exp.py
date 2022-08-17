"""
Plot a 3D graph of a single conditional probability
i.e., P(A2 | B), using the probability/test/models/nCondition.py SEM file.
Automatically generates data for each run.
Plots two graphs, comparing different variations of ProbSpace
cMethod parameter.  See ProbSpace for details on the different
methods.
Set the method using prob1 and prob2 constructors below.
"""
from sys import argv, path
if '.' not in path:
    path.append('.')
import time
from math import log, tanh, sqrt, sin, cos, e

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from because.synth import gen_data
from because.probability.rkhs.rkhsMV import RKHS
from because.visualization import grid2 as grid
from because.probability.prob import ProbSpace

def show(dataPath='', numRecs=0, targetSpec=[], condSpec=[], controlFor=[], gtype='pdf', probspace=None, enhance=False, power=1):
    assert len(targetSpec) == 1 and len(condSpec) == 1, 'probPlot2D_exp.show:  Must provide exactly one target and one condition.  Got: ' + str(targetSpec) + ', ' + str(condSpec)
    lim = 1
    dims = 2
 
    v1 = targetSpec[0][0]
    v2 = condSpec[0][0]

    #print('dims, datSize = ', dims, datSize)
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

    numRecs = prob1.N
    if numRecs <= 1000:
        numPts = 15
    elif numRecs <= 10000:
        numPts = 20
    elif numRecs < 100000:
        numPts = 25
    else:
        numPts = 30 # How many eval points for each conditional

    print('Dimensions = ', dims, '.  Conditionals = ', dims - 1)
    print('Number of points to test for each conditional = ', numPts)
    target = v1
    cond = v2 

    vars = [target, cond]
    g = grid.Grid(prob1, [v2], lim, numPts)
    tps = g.makeGrid()

    xt1 = []
    yt1 = []
    yt1_h = []
    yt1_l = []
    yt1_hh = []
    yt1_ll = []

    ptiles = [16, 2.5]
    dp_start = time.time()
    for tp in tps:
        bspec = tp[0]
        bnom = bspec[0]
        bquery = (cond,) + bspec[1:]
        if prob1.isStringVal(cond):
            bnom = prob1.numToStr(cond, bnom)
        condspec = [bquery]
        if controlFor:
            condspec = condspec + controlFor
        #print('condspec = ', condspec)
        #print('bval, bincr = ', bval, bincr)
        ey_x = prob1.E(target, condspec, power=power)
        dist = prob1.distr(target, condspec, power=power)
        #ey_x = dist.E()
        if dist is None or dist.N < 2:
            continue
        upper = dist.percentile(100-ptiles[0])
        lower = dist.percentile(ptiles[0])
        upper2 = dist.percentile(100-ptiles[1])
        lower2 = dist.percentile(ptiles[1])
        #print('ey_x, upper, lower, upper2, lower2 = ', ey_x, upper, lower, upper2, lower2)
        if ey_x is None:
            continue
        xt1.append(bnom)
        yt1.append(ey_x)
        yt1_h.append(upper)
        yt1_l.append(lower)    
        yt1_hh.append(upper2)
        yt1_ll.append(lower2)    
    dp_end = time.time()
    print('Elapsed Time = ', round(dp_end-dp_start, 3))
    fig = plt.figure(tight_layout = True)

    def sortByExp(x, y, y_h, y_l, y_hh, y_ll):
        allvar = [(y[i], x[i], y_h[i], y_l[i], y_hh[i], y_ll[i]) for i in range(len(x))]
        allvar.sort()
        allvar.reverse()
        y = [tup[0] for tup in allvar]
        x = [tup[1] for tup in allvar]
        y_h = [tup[2] for tup in allvar]
        y_l = [tup[3] for tup in allvar]
        y_hh = [tup[4] for tup in allvar]
        y_ll = [tup[5] for tup in allvar]
        return (x, y, y_h, y_l, y_hh, y_ll)
    x = np.array(xt1)
    y = np.array(yt1)
    y_h = np.array(yt1_h)
    y_l = np.array(yt1_l)
    y_hh = np.array(yt1_hh)
    y_ll = np.array(yt1_ll)
    if prob1.isCategorical(cond):
        x, y, y_h, y_l, y_hh, y_ll = sortByExp(x, y, y_h, y_l, y_hh, y_ll)

    my_cmap = plt.get_cmap('tab20')
    ax = fig.add_subplot(111)
    v2Label = v2
    v1Label = 'E( ' + v1 + ' | ' + v2 +' )'
    linecolor = my_cmap.colors[0]

    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_linewidth(1.5)
    ax.set_ylabel(v1Label, fontweight='bold', rotation = 90)
    ax.set_xlabel(v2Label, fontweight='bold', rotation = 0)
    ax.fill_between(x, y_hh, y_ll, color=linecolor, alpha=.1, linewidth=0)
    ax.fill_between(x, y_h, y_l, color=linecolor, alpha=.1, linewidth=0)
    ax.plot(x, y_hh, color=linecolor, alpha=.3, linewidth=.75)
    ax.plot(x, y_ll, color=linecolor, alpha=.3, linewidth=.75)
    ax.plot(x, y_h, color=linecolor, alpha=.4, linewidth=1)
    ax.plot(x, y_l, color=linecolor, alpha=.4, linewidth=1)
    ax.plot(x, y, color=linecolor, linewidth=3)
    plt.xlim([x[0], x[-1]])
    plt.ylim([np.min(y_ll), np.max(y_hh) * 1.01])
    plt.yticks(weight='bold')
    plt.xticks(rotation = -45, weight='bold')

    plt.show()



if __name__ == '__main__':
    if '-h' in argv or len(argv) < 4:
        print('Usage: python because/visualization/probPlot2D_exp.py dataPath targets condition [numRecs]')
        print('  dataPath is the path to a .py (synthetic data) or .csv file')
        print('  targets is the variable(s) whose distribution to plot.')
        print('  conditions are the conditional variable names.')
        print('  numRecs is the number of records to generate')
    else:
        numRecs = 0 
        args = argv
        dataPath = args[1].strip()
        targets = args[2].strip()
        conditions = args[3].strip()
        tokens = targets.split(',')
        tSpec = []
        for token in tokens:
            varName = token.strip()
            if varName:
                tSpec.append((varName,))
        tokens = conditions.split(',')
        cSpec = []
        for token in tokens:
            varName = token.strip()
            if varName:
                cSpec.append((varName,))
        if len(args) > 4:
            try:
                numRecs = int(args[4].strip())
            except:
                pass
        gtype = 'pdf'
       
        #print('dims, datSize, numRecs = ', dims, datSize, numRecs)
        show(dataPath=dataPath, numRecs=numRecs, targetSpec=tSpec, condSpec=cSpec, gtype=gtype)
