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

from because.causality import rv
from because.synth import read_data, gen_data
from because.probability import independence
from because.probability.prob import ProbSpace
from because.probability.rkhs.rkhsMV import RKHS
from because.visualization import grid2 as grid

def show(dataPath='', numRecs=0, targetSpec=[], condSpec=[], controlFor=[], gtype='pdf', probspace=None, enhance=True, power=1):
    assert len(targetSpec) == 1 and len(condSpec) == 1 or len(targetSpec) == 2 and len(condSpec) == 0, \
        'probPlot2D.show:  Must provide one target and one condition or two targets and no conditions.  Got: ' + str(targetSpec) + ', ' + str(condSpec)

    lim = 1  # Percentile limit to show on graph (i.e. [percentile(lim), percentile(100-lim)])
    
    cumulative = False
    if gtype == 'cdf':
        cumulative = True

    if probspace is None:
        # Generate the data
        f = open(dataPath, 'r')
        exec(f.read(), globals())

        print('Testing: ', dataPath, '--', testDescript)

        gen = gen_data.Gen(dataPath)
        data = gen.getDataset(numRecs)

        prob1 = ProbSpace(data)

    else:
        prob1 = probspace
    
    print('Controlling for: ', controlFor, ', power = ', power)
    
    N = prob1.N
    numRecs = N
    if numRecs <= 1000:
        numPts = 15
    elif numRecs <= 10000:
        numPts = 20
    elif numRecs < 100000:
        numPts = 25
    else:
        numPts = 30 # How many eval points for each conditional
    targets = [spec[0] for spec in targetSpec]
    conds = [spec[0] for spec in condSpec]
    if len(targets) == 1:
        joint = False
    else:
        joint = True
    if gtype == 'cdf':
        cumulative = True
    start = time.time()
    vars = targets + conds
    g = grid.Grid(prob1, vars, lim, numPts)
    tps = g.makeGrid()

    xt1 = []
    yt1 = []
    zt1 = []
    nTests = g.getTestCount(0)
    nTests_j = g.getTestCount()
    threshFactor = .2
    thresh = threshFactor / nTests
    thresh_j = threshFactor / nTests_j
    def enhanceResults(val):
        if joint:
            if val < thresh_j:
                return 0
            return tanh(val / (thresh_j * 5))
        else:
            if val < thresh:
                return 0
            return tanh(val / (thresh * 5))

    dp_start = time.time()
    for tp in tps:
        aspec = tp[0]
        if len(aspec) > 2:
            anom, alow, ahigh = aspec
        else:
            anom, aval = aspec

        bspec = tp[1]
        if len(bspec) > 2:
            bnom, blow, bhigh = bspec
        else:
            bnom, bval = bspec
        if cumulative:
            if not joint:
                # Conditional CDF
                tSpec = [(targets[0],)]
                cSpec = [(conds[0], None, bnom)]
                cSpec = cSpec + controlFor
                d = prob1.distr(tSpec, cSpec, power=power)
                if d is None:
                    psy_x = 0
                else:
                    psy_x = d.P((None, anom))
            else:
                # Joint CDF
                tSpec = [(targets[0], None, anom), (targets[1], None, bnom)]
                tSpec = tSpec + controlFor
                psy_x = prob1.P(tSpec, power=power)
        else:
            if not joint:
                # Conditional PDF
                if prob1.isDiscrete(targets[0]):
                    tSpec = (targets[0], aval)
                else:
                    tSpec = (targets[0], alow, ahigh)
                if prob1.isDiscrete(conds[0]):
                    cSpec = [(conds[0], bval)]
                else:
                    cSpec = [(conds[0], blow, bhigh)]
                cSpec = cSpec + controlFor
                psy_x = prob1.P(tSpec, cSpec, power=power)
                #print('psy_x = ', psy_x, ', tSpec, cSpec = ', tSpec, cSpec)
            else:
                # Joint PDF
                if prob1.isDiscrete(targets[0]):
                    tSpec1 = (targets[0], aval)
                else:
                    tSpec1 = (targets[0], alow, ahigh)
                if prob1.isDiscrete(targets[1]):
                    tSpec2 = (targets[1], bval)
                else:
                    tSpec2 = (targets[1], blow, bhigh)
 
                tSpec = [tSpec1, tSpec2]
                cSpec = controlFor
                psy_x = prob1.P(tSpec, cSpec, power=power)
        if enhance:
            psy_x = enhanceResults(psy_x)
        if psy_x is None or psy_x == 0:
            continue
        if psy_x > 1:
            #print('got p > 1:  aval, bval = ', aval, bval, psy_x)
            psy_x = 1
        xt1.append(anom)
        yt1.append(bnom)
        zt1.append(psy_x)    
    dp_end = time.time()

    print('Test Time = ', round(dp_end-dp_start, 3))
    strMappings = 'String Value Mappings:\n'
    hasStrMappings = False
    smap = prob1.stringMap
    for var in targets + conds:
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

    fig = plt.figure()


    x = np.array(xt1)
    y = np.array(yt1)
    z = np.array(zt1)

    my_cmap = plt.get_cmap('plasma')
    ax = fig.add_subplot(111, projection='3d')

    v1Label = '$' + vars[0] + '$'
    v2Label = '$' + vars[1] + '$'
    if joint:
        conj = ', '
    else:
        conj = ' | '
    if cumulative:
        v3Label = '$CDF( ' + vars[0] + conj + vars[1] + ' )$'
    else:
        v3Label = '$P(' + vars[0] + conj + vars[1] + ')$'
    ax.set_xlabel(v1Label, fontweight='bold', rotation = 0)
    ax.set_ylabel(v2Label, fontweight='bold', rotation = 0)
    ax.set_zlabel(v3Label, fontweight='bold', rotation = 0)
    ax.plot_trisurf(x, y, z, cmap = my_cmap)

    plt.show()

if __name__ == '__main__':
    if '-h' in argv or len(argv) < 4:
        print('Usage: python because/visualization/probPlot2D.py dataPath targets condition [numRecs]')
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
        if 'cum' in args:
            gtype = 'cdf'
       
        #print('dims, datSize, numRecs = ', dims, datSize, numRecs)
        show(dataPath=dataPath, numRecs=numRecs, targetSpec=tSpec, condSpec=cSpec, gtype=gtype)
