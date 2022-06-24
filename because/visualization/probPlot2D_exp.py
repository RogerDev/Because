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
    minvs = [distr.percentile(1) for distr in distrs]
    maxvs = [distr.percentile(99) for distr in distrs]
    incrs = [(maxvs[i] - minvs[i]) / (numPts-1) for i in range(dims)]
    
    for i in range(numPts):

        minv2 = minvs[1]
        incr2 = incrs[1]
        tp = minv2 + i * incr2
        tps.append(tp)
    #print('tps = ', tps[:10])
    # Traces for plot
    # 1 = Actual Function, 2 = JPROB, 3 = ProbSpace 
    xt1 = []
    yt1 = []
    yt1_h = []
    yt1_l = []
    yt1_hh = []
    yt1_ll = []

    ptiles = [15, 1.4]
    #print('Testpoints = ', tps)
    tnum = 0
    ssTot = 0 # Total sum of squares for R2 computation
    dp_est = []
    dp_start = time.time()
    for t in tps:
        bval = t
        bincr = incrs[1]
        condspec = (cond, bval, bval+bincr)
        ey_x = prob1.E(target, condspec)
        dist = prob1.distr(target, condspec)
        std = dist.stDev()
        #upper = ey_x + std
        #lower = ey_x - std
        upper = dist.percentile(100-ptiles[0])
        lower = dist.percentile(ptiles[0])
        upper2 = dist.percentile(100-ptiles[1])
        lower2 = dist.percentile(ptiles[1])
        if ey_x is None:
            continue
        xt1.append(t)
        yt1.append(ey_x)
        yt1_h.append(upper)
        yt1_l.append(lower)    
        yt1_hh.append(upper2)
        yt1_ll.append(lower2)    
    dp_end = time.time()
    results = []
    ysum = 0.0

fig = plt.figure()


x = np.array(xt1)
y = np.array(yt1)
y_h = np.array(yt1_h)
y_l = np.array(yt1_l)
y_hh = np.array(yt1_hh)
y_ll = np.array(yt1_ll)

my_cmap = plt.get_cmap('tab20')
ax = fig.add_subplot(111)
v2Label = '$' + v2 + '$'
v1Label = '$E( ' + v1 + ' | ' + v2 +' )$'
gray = my_cmap.colors[15]
dkgray = my_cmap.colors[14]
blue = my_cmap.colors[0]
yellow = my_cmap.colors[1]
ax.set_ylabel(v1Label, fontweight='bold', rotation = 90)
ax.set_xlabel(v2Label, fontweight='bold', rotation = 0)
ax.fill_between(x, y_hh, y_ll, color=gray, alpha=.3, linewidth=0)
ax.fill_between(x, y_h, y_l, color=gray, alpha=.2, linewidth=0)
ax.plot(x, y_hh, color=dkgray, alpha=.2, linewidth=.75)
ax.plot(x, y_ll, color=dkgray, alpha=.2, linewidth=.75)
ax.plot(x, y_h, color=dkgray, alpha=.2, linewidth=1)
ax.plot(x, y_l, color=dkgray, alpha=.2, linewidth=1)
ax.plot(x, y, color=blue, linewidth=2)

plt.show()