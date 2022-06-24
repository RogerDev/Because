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
dims = 1
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
if 'cum' in sys.argv:
    cumulative = True
numPts = 100
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

    lim = .1  # Percentile at which to start.  Will end at percentile(100-lim)

    print('Test Limit = ', lim, 'percentile to' , 100-lim, 'percentile.')
    N = prob1.N
    # P(A2 | B)
    target = v1
    evaluations = 0
    start = time.time()
    results = []
    vars = [v1]
    tps = []
    numTests = numPts**(dims)
    evaluations = 0
    mean = prob1.E(v1)
    dist = prob1.distr(v1)
    std = dist.stDev()
    skew = dist.skew()
    kurt = dist.kurtosis()
    minv = dist.percentile(lim)
    maxv = dist.percentile(100-lim)
    minv = dist.minVal()
    maxv = dist.maxVal()
    incr =  (maxv - minv) / (numPts-1)
    pl = dist.percentile(2.5)
    pll = dist.percentile(16)
    ph = dist.percentile(97.5)
    phh = dist.percentile(84)

    for i in range(numPts):
        tp = minv + incr * i
        tps.append(tp)
    #print('tps = ', tps[:10])
    # Traces for plot
    # 1 = Actual Function, 2 = JPROB, 3 = ProbSpace 
    xt1 = []
    xt2 = []
    yt1 = []
    yt2 = []

    target = v1
    #print('Testpoints = ', tps)
    tnum = 0
    ssTot = 0 # Total sum of squares for R2 computation
    dp_est = []
    dp_start = time.time()
    for t in tps:
        if cumulative:
            d = prob1.distr(target)
            if d is None:
                py = 0
            else:
                py = d.P((None, t))
        else:
           py = prob1.P((target, t, t+incr))
        if py is None:
            continue
        if py > 1:
            print('got p > 1:  t = ', t)
        else:
            xt1.append(t)
            yt1.append(py)
    dp_end = time.time()
    results = []
    ysum = 0.0

fig = plt.figure(tight_layout=True)


x = np.array(xt1)
y = np.array(yt1)

my_cmap = plt.get_cmap('tab20')
ax = fig.add_subplot(111)
if cumulative:
    v1Label = '$P( 0 <= '  + v1 + ' < X )$'
else:
    v1Label = '$P(' + v1 + '=X)$'
v2Label = '$' + 'X' + '$'
colors = my_cmap.colors
blue = colors[0]
gray = colors[15]
dkgray = colors[14]
green = colors[5]
yellow = colors[17]
ax.set(title = "Mean = " + str(round(mean, 3)) + ', std = ' + str(round(std, 3)) + ', skew = ' + str(round(skew, 3)) + ', kurt = ' + str(round(kurt, 3)))
ax.set_ylabel(v1Label, fontweight='bold', rotation = 90)
ax.set_xlabel(v2Label, fontweight='bold', rotation = 0)
ax.axvspan(phh, pll, color=gray, alpha=.1)
ax.axvspan(ph, pl, color=gray, alpha=.1)
plt.axvline(x=ph, color=yellow, alpha=1, linewidth = 1)
plt.axvline(x=pl, color=yellow, alpha=1, linewidth = 1)
plt.axvline(x=phh, color=yellow, alpha=1, linewidth = .75)
plt.axvline(x=pll, color=yellow, alpha=1, linewidth = .75)
plt.axvline(x=mean, color=green, linewidth = 2)
ax.axvspan(phh, pll, color=gray, alpha=.2)
ax.axvspan(ph, pl, color=gray, alpha=.2)
ax.fill_between(x, y, 0, color=blue, alpha=.3)
ax.plot(x, y, color=blue, linewidth=2)

plt.show()