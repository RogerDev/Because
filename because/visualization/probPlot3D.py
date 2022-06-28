"""
    Present a plot of the distributions for the given .py test file
    python3 Probabiity/probPlot.py <testfilepath>.py
    Data should previously have been generated using:
    python3 synth/synthDataGen.py <testfilepath>.py <numRecs>
"""
from sys import argv, path
if '.' not in path:
    path.append('.')
from because.synth import gen_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from because.probability.prob import ProbSpace
from because.visualization import grid
import numpy as np
from matplotlib import cm
import math

def show(dataPath='', numRecs=0, targetSpec=[], condSpec=[], gtype='pdf', probspace=None):
    assert (len(targetSpec) == 1 and len(condSpec) == 2) or (len(targetSpec) == 3 and len(condSpec) == 0), \
        'probPlot3D.show:  Must provide one target and two conditions or three targets and no conditions.  Got: ' + str(targetSpec) + ', ' + str(condSpec)
    power = 2
    lim = 1
    joint = False
    if len(targetSpec) == 3:
        joint = True
    cumulative = False
    if gtype == 'cdf':
        cumulative = True
    if numRecs <= 1000:
        dimPoints = 10
    elif numRecs <= 10000:
        dimPoints = 15
    elif numRecs < 500000:
        dimPoints = 20
    elif numRecs < 1000000:
        dimPoints = 25
    else:
        dimPoints = 30 # How many eval points for each conditional

    if probspace is None:
        f = open(dataPath, 'r')
        exec(f.read(), globals())

        print('Testing: ', dataPath, '--', testDescript)
        print('points per dimension = ', dimPoints)

        # For dat file, use the input file name with the .csv extension
        gen = gen_data.Gen(dataPath)
        data = gen.getDataset(numRecs)

        prob1 = ProbSpace(data, power=power)
    else:
        prob1 = probspace

    tVars = [spec[0] for spec in targetSpec]
    cVars = [spec[0] for spec in condSpec]
    vars = tVars + cVars
    g = grid.Grid(prob1, vars, lim, dimPoints)
    tps = g.makeGrid()
    incrs = g.getIncrs()


    xt = []
    yt = []
    zt = []
    my_cmap = plt.get_cmap('gray')

    tests = dimPoints**3 # Number of tests

    probs = []
    testNum = 1
    for tp in tps:
        aval = tp[0]
        bval = tp[1]
        cval = tp[2]
        aincr = incrs[0]
        bincr = incrs[1]
        cincr = incrs[2]
        if joint:
            if cumulative:
                targetSpec = [(vars[0], None, aval), (vars[1], None, bval), (vars[2], None, cval)]
            else:
                targetSpec = [(vars[0], aval, aval + aincr), (vars[1], bval, bval + bincr),
                                    (vars[2], cval, cval + cincr)]
            p = prob1.P(targetSpec)
            if p > 0:
                print(testNum, '/', tests, ': p = ', p)
                probs.append(p)
                xt.append(bval)
                yt.append(cval)
                zt.append(aval)
            testNum += 1
        else:
            if cumulative:
                targetSpec = (vars[0], None, aval)
                givensSpec = [(vars[1], None, bval), (vars[2], None, cval)]
            else:
                targetSpec = (vars[0], aval, aval + aincr)
                givensSpec = [(vars[1], bval, bval + bincr), (vars[2], cval, cval + cincr)]
            p = prob1.P(targetSpec, givensSpec)
            if p > .001 and p <= 1:
                print(testNum, '/', tests, ': p = ', p)
                probs.append(p)
                xt.append(bval)
                yt.append(cval)
                zt.append(aval)
            testNum += 1
    pltWidth = 200
    pltHeight = 150
    fig = plt.figure(constrained_layout=True)
    #fig = plt.figure(figsize=(pltWidth, pltHeight))
    x = np.array(xt)
    y = np.array(yt)
    z = np.array(zt)

    v1Label = '$' + vars[0] + '$'
    v2Label = '$' + vars[1] + '$'
    v3Label = '$' + vars[2] + '$'
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(v2Label, fontsize=20, rotation = 0)
    ax.set_ylabel(v3Label, fontsize=20, rotation = 0)
    ax.set_zlabel(v1Label, fontsize=20, rotation = 0)
    if joint:
        conj = ', '
    else:
        conj = ' | '
    if cumulative:
        ax.set(title = "CDF(" + vars[0] + conj + vars[1] + ", " + vars[2] + ")")
    else:
        ax.set(title = "P(" + vars[0] + conj + vars[1] + ", " + vars[2] + ")")

    N = len(x)

    def rescale(inProbs):
        minProb = min(inProbs)
        maxProb = max(inProbs)
        print('minProb, maxProb = ', minProb, maxProb)
        mean = np.mean(inProbs)
        probsS = inProbs.sort()
        median = inProbs[int(len(inProbs)/2)]
        outProbs = []
        for prob in inProbs:
            prob = prob / median / 2
            outProb = (math.tanh((prob-.5) * 2) + 1) / 2
            outProbs.append(outProb)
        return outProbs
    scaledProbs = rescale(probs)
    #print('minScaled, maxScaled = ', minScaled, maxScaled, scaledProbs[:1000])

    #colors = [my_cmap((1-prob))[:3] + (prob* 1,) for prob in scaledProbs]
    #colors = [(.5-prob/4, .5-prob/4, .7-prob/5) + (.2 + prob* .8,) for prob in scaledProbs]
    colors = [my_cmap(1-prob) for prob in scaledProbs]
    dotsize = np.array(scaledProbs) * (20000 / dimPoints)
    dotsize = 2000 / dimPoints
    ax.scatter(x, y, z, c=colors, edgecolors='none', marker='o', s=dotsize, linewidth=0)
    #ax.set_xlim3d(v2min, v2max)
    #ax.set_ylim3d(v3min, v3max)
    #ax.set_zlim3d(v1min, v1max)
    plt.show()

if __name__ == '__main__':
    if '-h' in argv or len(argv) < 4:
        print('Usage: python because/visualization/probPlot3D.py dataPath targets condition [numRecs]')
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