"""
Plot a 3D graph of a single conditional probability
i.e., P(A2 | B), using the models/nCondition.py SEM file.
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
if len(sys.argv) > 1:
    datSize = int(sys.argv[1])
if len(sys.argv) > 2:
    smoothness = float(sys.argv[2])
if 'cum' in sys.argv:
    cumulative=True
print('dims, datSize, tries = ', dims, datSize, tries)

test = 'models/nCondition.py'

f = open(test, 'r')
exec(f.read(), globals())

print('Testing: ', test, '--', testDescript)

# For dat file, use the input file name with the .csv extension
tokens = test.split('.')
testFileRoot = str.join('.',tokens[:-1])
datFileName = testFileRoot + '.csv'
jp_results = []
ps_results = []
jp_run = []
ps_run = []
for i in range(tries):
    # Generate a new sample
    gen = gen_data.Gen(test)
    sdg = gen.generate(datSize)
    print('sdg = ', sdg)
    d = read_data.Reader(datFileName)
    data = d.read()


    prob1 = ProbSpace(data, cMethod = 'd!')
    prob2 = ProbSpace(data, cMethod = 'u')

    lim = 2  # Std's from the mean to test conditionals
    if not cumulative:
        numPts = 30 # How many eval points for each conditional
    else:
        numPts = 20
    print('Test Limit = ', lim, 'standard deviations from mean')
    print('Dimensions = ', dims, '.  Conditionals = ', dims - 1)
    print('Number of points to test for each conditional = ', numPts)
    N = prob1.N
    # P(A2 | B)
    cond = 'B' 
    target = 'A2'

    evaluations = 0
    start = time.time()
    results = []
    totalErr_jp = 0
    totalErr_dp = 0
    vars = [target, 'B']
    tps = []
    numTests = numPts**(dims)
    evaluations = 0
    means = [prob1.E(c) for c in vars]
    stds = [prob1.distr(c).stDev() for c in vars]
    minvs = [means[i] - stds[i] * lim for i in range(dims)]
    incrs = [(std * lim - std * -lim) / (numPts-1) for std in stds]
    print('means = ', means)
    print('stds = ', stds)
    
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
    cmprs = []
    jp_est = []
    dp_est = []


    #print('Testing JPROB')
    jp_start = time.time()
    for t in tps:
        aval = t[0]
        bval = t[1]
        condspec = ('B', bval)

        if cumulative:
            d = prob2.distr(target, condspec)
            if d is None:
                psy_x = 0
            else:
                psy_x = d.P((None, aval))
        else:
            d = prob2.distr(target, condspec)
            psy_x = d.P(aval)
        if psy_x is None:
            continue
        if psy_x > 1:
            print('got p > 1:  aval, bval = ', aval, bval)
        else:
            jp_est.append(psy_x)
            xt2.append(t[0])
            yt2.append(t[1])
            zt2.append(psy_x)    
    jp_end = time.time()
    #print('Testing PS')
    dp_start = time.time()
    for t in tps:
        #psy_x = prob.E(v1, [(v2, v2val - .1 * v2std, v2val + .1 * v2std) , (v3, v3val - .1 * v3std, v3val + .1 * v3std)])
        aval = t[0]
        bval = t[1]
        condspec = ('B', bval)

        if cumulative:
            d = prob1.distr(target, condspec)
            if d is None:
                psy_x = 0
            else:
                psy_x = d.P((None, aval))
        else:
            d = prob1.distr(target, condspec)
            psy_x = d.P(aval)
        dp_est.append(psy_x)
        xt3.append(t[0])
        yt3.append(t[1])
        zt3.append(psy_x)
    dp_end = time.time()
    #for i in range(len(tps)):
    #    print('tps, jp_est, ps_est = ', tps[i], jp_est[i], ps_est[i])
    totalErr_jp = 0.0
    totalErr_dp = 0.0
    results = []
    ysum = 0.0


    for result in results:
        pass
        #print('tp, y|X, ps, ref, err2_jp, err2_ps = ', result[0], result[1], result[2], result[3], result[4], result[5])

    #print('RMSE PS = ', rmse_ps)
    # Calc R2 for each

    #print('R2 JP =', R2_jp)
    #print('R2 PS =', R2_ps)

fig = plt.figure()

#fig = plt.figure()
x = np.array(xt2)
y = np.array(yt2)
z = np.array(zt2)
#print('x, y, z = ', len(xt2), len(yt2), len(zt2))
#print('z = ', z)
#xyz = [(x[i], y[i], z[i]) for i in range(len(x))]
#print('xyz = ', xyz)
#print('x, y, z 2 =', len(x), len(y), len(z))
my_cmap = plt.get_cmap('winter')
ax = fig.add_subplot(121, projection='3d')
ax.plot_trisurf(x, y, z, cmap = my_cmap)

x = np.array(xt3)
y = np.array(yt3)
z = np.array(zt3)
#xyz = [(x[i], y[i], z[i]) for i in range(len(x))]
#print('x, y, z 2 =', len(x), len(y), len(z))
ax = fig.add_subplot(122, projection='3d')
ax.plot_trisurf(x, y, z, cmap = my_cmap)

#ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5);
plt.show()