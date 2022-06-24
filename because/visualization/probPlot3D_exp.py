"""
    Present a plot of the distributions for the given .py test file
    python3 Probabiity/probPlot.py <testfilepath>.py
    Data should previously have been generated using:
    python3 synth/synthDataGen.py <testfilepath>.py <numRecs>
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
power = 3
lim = 2  # Std's from the mean to test conditionals
numPts = 30 # How many eval points for each conditional
 
# Arg format is  <datSize>
dims = 3
smoothness = .8
cumulative = False
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
if len(args) > 5:
    v3 = args[5].strip()
else:
    v3 = 'A'
joint = False
if 'joint' in sys.argv:
    joint = True
print('dims, datSize, tries = ', dims, datSize, tries)
if datSize <= 1000:
    numPts = 15
elif datSize <= 10000:
    numPts = 20
elif datSize < 100000:
    numPts = 25
else:
    numPts = 30 # How many eval points for each conditional

#test = 'probability/test/models/doubleCondition.py'
test = 'probability/test/models/nCondition.py'

f = open(test, 'r')
exec(f.read(), globals())

print('Testing: ', test, '--', testDescript)

# For dat file, use the input file name with the .csv extension

tokens = test.split('.')
testFileRoot = str.join('.',tokens[:-1])
datFileName = testFileRoot + '.csv'
jp_results = []
dp_results = []
up_results = []
jp_run = []
dp_run = []
up_run = []
for i in range(tries):
    gen = gen_data.Gen(test)
    data = gen.getDataset(datSize)
    prob1 = ProbSpace(data, cMethod='d!', power=power)

    print('Test Limit = ', lim, 'standard deviations from mean')
    print('Dimensions = ', dims, '.  Conditionals = ', dims - 1)
    print('Number of points to test for each conditional = ', numPts)
    N = prob1.N
    #evalpts = int(sqrt(N)) # How many target points to sample for expected value:  E(Z | X=x. Y=y)
    #print('JPROB points for mean evaluation = ', evalpts)
    vars = prob1.fieldList
    target = v1
    cond = [v2, v3]

    amean = prob1.E(target)
    astd = prob1.distr(target).stDev()
    amin = amean - lim * astd
    arange = lim * astd - lim * -astd
    #aincr = arange / (evalpts - 1)
    #print('A: mean, std, range, incr = ', amean, astd, arange, aincr)
    #R1 = RKHS(prob.ds, delta=None, includeVars=[target] + cond[:dims-1], s=smoothness)
    #R2 = RKHS(prob.ds, delta=None, includeVars=cond[:dims-1], s=smoothness)
    evaluations = 0
    start = time.time()
    results = []
    totalErr_jp = 0
    totalErr_dp = 0
    totalErr_up = 0
    conds = len(cond)
    tps = []
    numTests = numPts**(dims-1)
    evaluations = 0
    distrs = [prob1.distr(c) for c in cond]
    means = [distr.E() for distr in distrs]
    stds = [distr.stDev() for distr in distrs]
    minvs = [distr.percentile(1) for distr in distrs]
    maxvs = [distr.percentile(99) for distr in distrs]
    incrs = [(maxvs[i] - minvs[i]) / (numPts-1) for i in range(conds)]

    # Generate the test points
    for i in range(numTests):
        tp = []
        for j in range(dims-1):
            minv = minvs[j]
            incr = incrs[j]
            mod = numPts**(dims - 1 - j - 1)
            #print('mod = ', mod, j)
            p = minv + int(i/mod)%numPts * incr
            tp.append(p)
        tps.append(tp)
    # Traces for plot
    # 1 = Actual Function, 2 = JPROB, 3 = ProbSpace 
    xt1 = []
    yt1 = []
    zt1 = []
    #print('Testpoints = ', tps)
    tnum = 0
    ssTot = 0 # Total sum of squares for R2 computation
    cmprs = []
    dp_est = []
    for t in tps:
        condspec = []
        for c in range(dims-1):
            condVar = cond[c]
            val = t[c]
            spec = (condVar, val, val+incrs[c])
            condspec.append(spec)
        y_x = prob1.E(target, condspec)
        if y_x is None:
            continue
        xt1.append(t[0])
        yt1.append(t[1])
        zt1.append(y_x)
    dp_end = time.time()
fig = plt.figure(constrained_layout=True)
fig.suptitle('N=' + str(datSize))
x = np.array(xt1)
y = np.array(yt1)
z = np.array(zt1)
my_cmap = plt.get_cmap('winter')
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(x, y, z, cmap = my_cmap)
ax.set_xlabel(v2, fontweight='bold')
ax.set_ylabel(v3, fontweight='bold')
ax.set_zlabel('E(' + v1 + ' | ' + v2 + ', ' + v3 + ')', fontweight='bold')
ax.view_init(20, -165)

plt.show()