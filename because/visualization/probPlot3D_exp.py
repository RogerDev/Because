"""
    Present a plot of the distributions for the given .py test file
    python3 Probabiity/probPlot.py <testfilepath>.py
    Data should previously have been generated using:
    python3 synth/synthDataGen.py <testfilepath>.py <numRecs>
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

from because.causality import rv
from because.synth import read_data, gen_data
from because.probability import independence
from because.probability.prob import ProbSpace
from because.probability.rkhs.rkhsMV import RKHS
from because.visualization import grid2 as grid

def show(dataPath='', numRecs=0, targetSpec=[], condSpec=[], controlFor=[], gtype='pdf', probspace=None, power=1, enhance=False):
    assert len(targetSpec) == 1 and len(condSpec) == 2, 'probPlot3D_exp.show:  Must provide exactly one target and two conditions.  Got: ' + str(targetSpec) + ', ' + str(condSpec)

    lim = 1  # Percentile limit to show on graph (i.e. [percentile(lim), percentile(100-lim)])
    numPts = 20 # How many eval points for each conditional
    
    dims = 3

    if probspace is None:
        f = open(dataPath, 'r')
        exec(f.read(), globals())

        print('Testing: ', dataPath, '--', testDescript)
        gen = gen_data.Gen(dataPath)
        data = gen.getDataset(numRecs)
        prob1 = ProbSpace(data, power=power)
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

    print('numPts = ', numPts)
    target = targetSpec[0][0]
    cond = [condSpec[0][0], condSpec[1][0]]


    # Generate the test points
    g = grid.Grid(prob1, cond, lim, numPts)
    tps = g.makeGrid()
    nTests = g.getTestCount()
    print('nTests = ', nTests)

    xt1 = []
    yt1 = []
    zt1 = []

    dp_start = time.time()
    for t in tps:
        condspec = []
        noms = []
        for c in range(dims-1):
            condVar = cond[c]
            spec = t[c]
            cnom = spec[0]
            if prob1.isCategorical(condVar):
                cval = spec[1]
                spec = (condVar, cval)
            else:
                clow = spec[1]
                chigh = spec[2]
                spec = (condVar, clow, chigh)
            condspec.append(spec)
            noms.append(cnom)
        condspec2 = condspec + controlFor
        y_x = prob1.E(target, condspec2, power=power)
        #print('y_x = ', y_x, ', condspec2 = ', condspec2)
        if type(y_x) == type(''):
            # If a string (categorical) value, map it to a number.
            y_x = prob1.getNumValue(target, y_x)
        jp = prob1.P(condspec, power=power)
        if enhance and jp < .2 / nTests:
            continue
        if y_x is None:
            continue
        xt1.append(noms[0])
        yt1.append(noms[1])
        zt1.append(y_x)
    dp_end = time.time()
    print('Test Time = ', round(dp_end-dp_start, 3))
    strMappings = 'String Value Mappings:\n'
    hasStrMappings = False
    smap = prob1.stringMap
    for var in cond:
        if prob1.isStringVal(var):
            hasStrMappings = True
            map = smap[var]
            valTokens = []
            for key in map.keys():
                val = map[key]
                valTokens.append(key + '=' + str(val))
            valStr = ', '.join(valTokens)
            strMappings += '     ' + var + ': ' + valStr + '\n'
    if hasStrMappings:
        print(strMappings)
    fig = plt.figure(constrained_layout=False)
    x = np.array(xt1)
    y = np.array(yt1)
    z = np.array(zt1)
    my_cmap = plt.get_cmap('plasma')
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x, y, z, cmap = my_cmap)
    ax.set_xlabel(cond[0], fontweight='bold')
    ax.set_ylabel(cond[1], fontweight='bold')
    ax.set_zlabel('E(' + target + ' | ' + cond[0] + ', ' + cond[1] + ')', fontweight='bold')
    ax.view_init(20, -165)

    plt.show()

if __name__ == '__main__':
    if '-h' in argv or len(argv) < 4:
        print('Usage: python because/visualization/probPlot3D_exp.py dataPath targets condition [numRecs]')
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
