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
    lim = 1  # Percentile limit to show on graph (i.e. [percentile(lim), percentile(100-lim)])
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
        numPts = 20
    elif numRecs <= 10000:
        numPts = 25
    elif numRecs < 100000:
        numPts = 30
    else:
        numPts = 35 # How many eval points for each conditional

    print('Dimensions = ', dims, '.  Conditionals = ', dims - 1)
    print('Number of points to test for each conditional = ', numPts)
    target = v1
    cond = v2 

    g = grid.Grid(prob1, [v2], lim, numPts)
    tps = g.makeGrid()

    xt1 = []
    yt1 = []

    condIsString = prob1.isStringVal(cond)
    
    dp_start = time.time()
    for tp in tps:
        bspec = tp[0]
        bnom = bspec[0]
        bquery = (cond,) + bspec[1:]
        if condIsString:
            bval = prob1.numToStr(cond, bspec[1])
            bnom = bval
            condspec = (cond, bval)
        else:
            condspec = bquery
        if controlFor:
            condspec = [condspec] + controlFor
        #print('condspec = ', condspec)
        py_x = prob1.P(targetSpec, condspec, power=power)
        #print('ey_x, upper, lower, upper2, lower2 = ', ey_x, upper, lower, upper2, lower2)
        xt1.append(bnom)
        yt1.append(py_x)
    dp_end = time.time()
    print('Test Time = ', round(dp_end-dp_start, 3))
    fig = plt.figure(tight_layout=True)

    if condIsString:
        xy = [(yt1[i], xt1[i]) for i in range(len(yt1))]
        xy.sort()
        xy.reverse()
        xt1 = [item[1] for item in xy]
        yt1 = [item[0] for item in xy]
    
    x = np.array(xt1)
    y = np.array(yt1)

    
    my_cmap = plt.get_cmap('tab20')
    ax = fig.add_subplot(111)
    ax.grid(visible=False)
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_linewidth(1.5)
    v2Label = v2
    v1Label = 'P( ' + str(targetSpec) + ' | ' + v2 +' )'
    colors = my_cmap.colors
    import random
    colorInd = 0
    linecolor = colors[colorInd]
    #linecolor = (0,0,.4)
    ax.set_ylabel(v1Label, fontsize='medium', fontweight='bold', rotation = 90)
    ax.set_xlabel(v2Label, fontsize='large', fontweight='bold', rotation = 0)
    ax.plot(x, y, color=linecolor, alpha=.8, linewidth=2)
    ax.fill_between(x, y, 0, color=linecolor, alpha=.5)
    plt.ylim([.9 * np.min(y), 1.1 * np.max(y)])
    plt.xlim([x[0], x[-1]])
    plt.yticks(weight='bold')
    plt.xticks(rotation = -45, weight='bold')
    plt.show()



if __name__ == '__main__':
    if '-h' in argv or len(argv) < 4:
        print('Usage: python because/visualization/probPlot2D_bound.py dataPath targets condition [numRecs]')
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
