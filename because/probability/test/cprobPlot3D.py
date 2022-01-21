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
# Arg format is  <datSize>
dims = 3
smoothness = 1
cumulative = False
if len(sys.argv) > 1:
    datSize = int(sys.argv[1])
if len(sys.argv) > 2:
    smoothness = float(sys.argv[2])
print('dims, datSize, tries = ', dims, datSize, tries)

test = 'models/doubleCondition.py'

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
    sdg = gen.generate(datSize)
    d = read_data.Reader(datFileName)
    data = d.read()


    prob1 = ProbSpace(data, cMethod='d!')
    prob2 = ProbSpace(data, cMethod='j')
    prob3 = ProbSpace(data, cMethod='u')

    lim = 3  # Std's from the mean to test conditionals
    numPts = 30 # How many eval points for each conditional
    print('Test Limit = ', lim, 'standard deviations from mean')
    print('Dimensions = ', dims, '.  Conditionals = ', dims - 1)
    print('Number of points to test for each conditional = ', numPts)
    N = prob1.N
    #evalpts = int(sqrt(N)) # How many target points to sample for expected value:  E(Z | X=x. Y=y)
    #print('JPROB points for mean evaluation = ', evalpts)
    vars = prob1.fieldList
    cond = []
    # Get the conditional variables
    for i in range(len(vars)):
        var = vars[i]
        if var[0] != 'A':
            cond.append(var)

    target = 'A'

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
    means = [prob1.E(c) for c in cond]
    stds = [prob1.distr(c).stDev() for c in cond]
    minvs = [means[i] - stds[i] * lim for i in range(len(means))]
    incrs = [(std * lim - std * -lim) / (numPts-1) for std in stds]

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
    xt2 = []
    xt3 = []
    xt4 = []
    yt1 = []
    yt2 = []
    yt3 = []
    yt4 = []
    zt1 = []
    zt2 = []
    zt3 = []
    zt4 = []

    #print('Testpoints = ', tps)
    tnum = 0
    ssTot = 0 # Total sum of squares for R2 computation
    cmprs = []
    jp_est = []
    dp_est = []
    up_est = []
    # Generate the target values for comparison
    for t in tps:
        cmpr = tanh(t[0]) + sin(t[1])
        cmprs.append(cmpr)
        xt1.append(t[0])
        yt1.append(t[1])
        zt1.append(cmpr)

    #print('Testing JPROB')
    jp_start = time.time()
    for t in tps:
        condspec = []
        for c in range(dims-1):
            condVar = cond[c]
            val = t[c]
            spec = (condVar, val)
            condspec.append(spec)
        y_x = prob2.E(target, condspec)

        jp_est.append(y_x)
        xt2.append(t[0])
        yt2.append(t[1])
        zt2.append(y_x)
    jp_end = time.time()
    #print('Testing UPROB')
    up_start = time.time()
    for t in tps:
        condspec = []
        for c in range(dims-1):
            condVar = cond[c]
            val = t[c]
            spec = (condVar, val)
            condspec.append(spec)
        y_x = prob3.E(target, condspec)

        up_est.append(y_x)
        xt4.append(t[0])
        yt4.append(t[1])
        zt4.append(y_x)
    up_end = time.time()
    #print('Testing dp')
    dp_start = time.time()
    for t in tps:
        condspec = []
        for c in range(dims-1):
            condVar = cond[c]
            val = t[c]
            spec = (condVar, val)
            condspec.append(spec)
        y_x = prob1.E(target, condspec)
        dp_est.append(y_x)
        xt3.append(t[0])
        yt3.append(t[1])
        zt3.append(y_x)
    dp_end = time.time()
    totalErr_jp = 0.0
    totalErr_dp = 0.0
    results = []
    ysum = 0.0
    for i in range(len(cmprs)):
        t = tps[i]
        cmpr = cmprs[i]
        ysum += cmpr
        jp_e = jp_est[i]
        dp_e = dp_est[i]
        up_e = up_est[i]
        error2_jp = (cmpr-jp_e)**2
        error2_dp = (cmpr-dp_e)**2
        error2_up = (cmpr-up_e)**2
        totalErr_jp += error2_jp
        totalErr_dp += error2_dp
        totalErr_up += error2_up
        results.append((t, jp_e, dp_e, up_e, cmpr, error2_jp, error2_dp, error2_up))

    for result in results:
        pass
        #print('tp, y|X, dp, ref, err2_jp, err2_dp = ', result[0], result[1], result[2], result[3], result[4], result[5])
    rmse_jp = sqrt(totalErr_jp) / len(tps)

    rmse_dp = sqrt(totalErr_dp) / len(tps)
    rmse_dp = sqrt(totalErr_dp) / len(tps)
    rmse_up = sqrt(totalErr_up) / len(tps)
    # Calc R2 for each
    yavg = ysum / len(tps)
    ssTot = sum([(c - yavg)**2 for c in cmprs])
    R2_jp = 1 - totalErr_jp / ssTot
    R2_dp = 1 - totalErr_dp / ssTot
    R2_up = 1 - totalErr_up / ssTot
    print('JP:')
    print('   R2 =', R2_jp)
    print('DP:')
    print('   R2 =', R2_dp)
    print('UP:')
    print('   R2 =', R2_up)
    jp_runtime = round(jp_end - jp_start, 5)
    dp_runtime = round(dp_end - dp_start, 5)
    up_runtime = round(up_end - up_start, 5)
 
    jp_results.append(R2_jp)
    dp_results.append(R2_dp)
    up_results.append(R2_up)
    jp_run.append(jp_runtime)
    dp_run.append(dp_runtime)
    up_run.append(up_runtime)
jp_avg = np.mean(jp_results)
dp_avg = np.mean(dp_results)
up_avg = np.mean(up_results)
jp_min = np.min(jp_results)
dp_min = np.min(dp_results)
up_min = np.min(up_results)
jp_max = np.max(jp_results)
dp_max = np.max(dp_results)
up_max = np.max(up_results)
jp_std = np.std(jp_results)
dp_std = np.std(dp_results)
up_std = np.std(up_results)
jp_runt = np.mean(jp_run)
dp_runt = np.mean(dp_run)
up_runt = np.mean(up_run)
#error = min(max(0, (dp_avg - jp_avg)/ dp_avg), 1)
print('dims, datSize, tries = ', dims, datSize, tries)
print('Average R2: JP, DP, UP = ', jp_avg, dp_avg, up_avg)
print('Min R2: JP, DP, UP = ', jp_min, dp_min, up_min)
print('Max R2: JP, DP, UP = ', jp_max, dp_max, up_max)
print('Std R2: JP, DP, UP = ', jp_std, dp_std, up_std)
print('Runtimes: JP, DP, UP = ', jp_runt, dp_runt, up_runt)
print('NumTests = ', tries)
# Ideal
fig = plt.figure(constrained_layout=True)
fig.suptitle('N=' + str(datSize))
x = np.array(xt1)
y = np.array(yt1)
z = np.array(zt1)
print(x[:10])
print(z[:10])
my_cmap = plt.get_cmap('winter')
ax = fig.add_subplot(141, projection='3d')
ax.plot_trisurf(x, y, z, cmap = my_cmap)
ax.set_xlabel('X', fontweight='bold')
ax.set_ylabel('Y', fontweight='bold')
ax.set_zlabel('E(Z|X,Y)', fontweight='bold')
ax.set(title = "Ideal")
ax.view_init(20, -165)

# J-Prob
x = np.array(xt2)
y = np.array(yt2)
z = np.array(zt2)
#print('x, y, z 2 =', len(x), len(y), len(z))
ax = fig.add_subplot(143, projection='3d')
ax.plot_trisurf(x, y, z, cmap = my_cmap)
ax.set_xlabel('X', fontweight='bold')
ax.set_ylabel('Y', fontweight='bold')
ax.set_zlabel('E(Z|X,Y)', fontweight='bold')
ax.set(title = "J-Prob  (R2 = " + str(round(jp_avg,2)) + ")")
ax.view_init(20, -165)

# D-Prob
x = np.array(xt3)
y = np.array(yt3)
z = np.array(zt3)
ax = fig.add_subplot(142, projection='3d')
ax.plot_trisurf(x, y, z, cmap = my_cmap)
ax.set_xlabel('X', fontweight='bold')
ax.set_ylabel('Y', fontweight='bold')
ax.set_zlabel('E(Z|X,Y)', fontweight='bold')
ax.set(title = "D-Prob (R2 = " + str(round(dp_avg,2)) + ")")
ax.view_init(20, -165)

# U-Prob
x = np.array(xt4)
y = np.array(yt4)
z = np.array(zt4)
ax = fig.add_subplot(144, projection='3d')
ax.plot_trisurf(x, y, z, cmap = my_cmap)
ax.set_xlabel('X', fontweight='bold')
ax.set_ylabel('Y', fontweight='bold')
ax.set_zlabel('E(Z|X,Y)', fontweight='bold')
ax.set(title = "U-Prob (R2 = " + str(round(up_avg,2)) + ")")
ax.view_init(20, -165)

plt.show()