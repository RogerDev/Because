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
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from because.causality import rv
from because.synth import read_data, gen_data
from because.probability import independence
from because.probability.prob import ProbSpace
from because.probability.rkhs.rkhsMV import RKHS
from because.visualization import grid

def show(dataPath='', numRecs=0, targetSpec=[], condSpec=[], controlFor=[], gtype='pdf', probspace=None, enhance=True, power=0):

    if probspace is None:
        # Got a .csv or .py file
        tokens = dataPath.split('.')
        ds = None # The dataset in dictionary form
        assert len(tokens) == 2 and (tokens[1] == 'py' or tokens[1] == 'csv'), 'Heatmap.show: dataPath must have a .py or .csv extension.  Got: ' + dataPath
        if tokens[1] == 'py':
            # py SEM file
            assert numRecs > 0, 'Heatmap.show: For synthetic data (i.e. .py extension) numRecs must be positive'
            gen = gen_data.Gen(dataPath)
            ds = gen.getDataset(numRecs)
        else:
            # csv
            r = read_data.Reader(dataPath)
            ds = r.read()
        prob1 = ProbSpace(ds, power=power)
    else:
        prob1 = probspace
    if power == 0:
        print('Heatmap.show: Showing Correlation Coefficients (power=0).')
        print('  Positive values indicate positive correlation.')
        print('  Negative values indicate negative correlation.  Zero indicate no correlation.')
    else:
        print('Heatmap.show: Showing dependence between variables with sensitivity = ', power, '.')
        print('  Values near 1.0 indicate strong dependence.  Values near 0.0 indicate strong independence.')
        print('  Values near .5 are indeterminate.')
    if not targetSpec:
        vars = prob1.getVarNames()
        vars.sort()
    else:
        vars = [spec[0] for spec in targetSpec]
    
    corr = []
    corrCache = {}
    for i in range(len(vars)):
        row = []
        v1 = vars[i]
        for j in range(len(vars)):
            v2 = vars[j]
            if i == j:
                row.append(1.0)
            elif i > j:
                # We already processed the reverse.  Use that from cache.
                dep = corrCache[(v2, v1)]
                row.append(dep)
            else:
                #print(v1, v2)
                if power == 0:
                    dep = prob1.corrCoef(v1,v2)
                else:
                    dep = prob1.dependence(v1,v2, power=power, sensitivity=power)
                row.append(dep)
                corrCache[(v1, v2)] = dep
        corr.append(row)
    

    cmap = 'BrBG'
    fig, ax = plt.subplots()
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    im = ax.imshow(corr, cmap=cmap, norm=norm)
    threshold = .5
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(vars)), labels=vars)
    ax.set_yticks(np.arange(len(vars)), labels=vars)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Create gray grid to separate squares.

    ax.set_xticks(np.arange(len(vars)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(vars)+1)-.5, minor=True)
    ax.grid(which="minor", color=(.5,.5,.5), linestyle='-', linewidth=.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    kw = dict(horizontalalignment="center", verticalalignment="center")
    # Loop over data dimensions and create text annotations.
    for i in range(len(vars)):
        for j in range(len(vars)):
            if i == j:
                continue
            val = round(corr[i][j], 2)
            kw.update(color = 'white' if abs(val) > threshold else 'black')
            text = ax.text(j, i, str(val), **kw)

    if power == 0:
        ax.set_title("Variable Correlation Heat Map")
    else:
        ax.set_title("Variable Dependence Heat Map")
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    if '-h' in argv or len(argv) < 2:
        print('Usage: python because/visualization/heatmap.py dataPath targets [numRecs]')
        print('  dataPath is the path to a .py (synthetic data) or .csv file')
        print('  targets is the comma separated list of variables to show.')
        print('  numRecs is the number of records to generate')
    else:
        numRecs = 0 
        args = argv
        dataPath = args[1].strip()
        targets = args[2].strip()
        tokens = targets.split(',')
        tSpec = []
        for token in tokens:
            varName = token.strip()
            if varName:
                tSpec.append((varName,))
        cSpec = []

        if len(args) > 2:
            try:
                numRecs = int(args[4].strip())
            except:
                pass
        gtype = 'pdf'
       
        #print('dims, datSize, numRecs = ', dims, datSize, numRecs)
        show(dataPath=dataPath, numRecs=numRecs, targetSpec=tSpec, condSpec=cSpec, gtype=gtype)
