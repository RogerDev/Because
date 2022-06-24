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

tries = 1
datSize = 200
# Arg format is  <datSize>
dims = 2
smoothness = 1
cumulative = False
if '-h' in sys.argv:
    print('Usage: python because/probability/test/cprobPlot2D.py datSize [smoothness] [cum]')
    print('  datSize is the number of records to generate')
    print('  smoothness is the smoothness parameter to use for kernel methods (default 1.0)')
    print('  "cum" as a final parameter causes comparison of cumulative distributions.')
args = sys.argv
if len(args) > 1:
    test = args[1]
if len(args) > 2:
    datSize = int(args[2].strip())
if len(args) > 3:
    v1 = args[3].strip()
else:
    v1 = 'C'
if len(args) > 4:
    v2 = args[4].strip()
else:
    v2 = 'B'
joint = False
if 'joint' in sys.argv:
    joint = True
elif 'cum' in sys.argv:
    cumulative=True
if datSize <= 1000:
    numPts = 15
elif datSize <= 10000:
    numPts = 20
elif datSize < 100000:
    numPts = 25
else:
    numPts = 30 # How many eval points for each conditional
print('dims, datSize, tries = ', dims, datSize, tries)

f = open(test, 'r')
exec(f.read(), globals())

print('Testing: ', test, '--', testDescript)

# Generate the data
jp_results = []
ps_results = []
jp_run = []
ps_run = []
for i in range(tries):
    # Generate a new sample
    gen = gen_data.Gen(test)
    data = gen.getDataset(datSize)

    prob1 = ProbSpace(data, cMethod = 'd!')
    prob2 = ProbSpace(data, cMethod = 'd!')

    print('Dimensions = ', dims, '.  Conditionals = ', dims - 1)
    print('Number of points to test for each conditional = ', numPts)
    N = prob1.N
    # P(A2 | B)
    target = v1
    cond = v2 

    evaluations = 0
    start = time.time()
    results = []
    vars = [target, cond]
    tps = []
    numTests = numPts**(dims)
    evaluations = 0
    distrs = [prob1.distr(c) for c in vars]
    means = [prob1.E(c) for c in vars]
    stds = [distr.stDev() for distr in distrs]
    minvs = [distr.minVal() for distr in distrs]
    maxvs = [distr.maxVal() for distr in distrs]
    incrs = [(maxvs[i] - minvs[i]) / (numPts-1) for i in range(dims)]
    
    for i in range(numPts):
        for j in range(numPts):
            minv1 = minvs[0]
            incr1 = incrs[0]
            minv2 = minvs[1]
            incr2 = incrs[1]
            tp = [minv1 + i * incr1, minv2 + j * incr2]
            tps.append(tp)
    #print('tps = ', tps[:10])
    # Traces for plot
    # 1 = Actual Function, 2 = JPROB, 3 = ProbSpace 
    xt1 = []
    xt2 = []
    xt3 = []
    yt1 = []
    yt2 = []
    yt3 = []
    zt1 = []
    zt2 = []
    zt3 = []

    #print('Testpoints = ', tps)
    tnum = 0
    ssTot = 0 # Total sum of squares for R2 computation
    dp_est = []
    dp_start = time.time()
    for t in tps:
        aval = t[0]
        bval = t[1]
        aincr = incrs[0]
        bincr = incrs[1]
        condspec = (cond, bval, bval+bincr)
        if cumulative:
            d = prob2.distr(target, condspec)
            if d is None:
                psy_x = 0
            else:
                psy_x = d.P((None, aval))
        elif joint:
            jTarg = [(target, aval, aval+aincr), (cond, bval, bval+bincr)]
            psy_x = prob2.P(jTarg)
        else:
            #d = prob2.distr(target, condspec)
            #if d is None:
            #    continue
            psy_x = prob2.P((target, aval-aincr, aval+aincr), condspec)
        if psy_x is None:
            continue
        if psy_x > 1:
            #print('got p > 1:  aval, bval = ', aval, bval, psy_x)
            psy_x = 1
        dp_est.append(psy_x)
        xt1.append(t[0])
        yt1.append(t[1])
        zt1.append(psy_x)    
    dp_end = time.time()
    results = []
    ysum = 0.0

fig = plt.figure()


x = np.array(xt1)
y = np.array(yt1)
z = np.array(zt1)

my_cmap = plt.get_cmap('winter')
ax = fig.add_subplot(111, projection='3d')
v1Label = '$' + v1 + '$'
v2Label = '$' + v2 + '$'
if joint:
    v3Label = '$P(' + v1 + ', ' + v2 + ')$'
else:
    v3Label = '$P(' + v1 + ' | ' + v2 + ')$'
ax.set_xlabel(v1Label, fontweight='bold', rotation = 0)
ax.set_ylabel(v2Label, fontweight='bold', rotation = 0)
ax.set_zlabel(v3Label, fontweight='bold', rotation = 0)
ax.plot_trisurf(x, y, z, cmap = my_cmap)

plt.show()