"""
    Present a plot of the distributions for the given .py test file
    python3 Probabiity/probPlot.py <testfilepath>.py
    Data should previously have been generated using:
    python3 synth/synthDataGen.py <testfilepath>.py <numRecs>
"""
import sys
if '.' not in sys.path:
    sys.path.append('.')
from synth import gen_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from probability.prob import ProbSpace
import numpy as np
from matplotlib import cm
import math




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
cumulative = False
if 'cum' in sys.argv:
    cumulative = True
if datSize <= 1000:
    dimPoints = 10
elif datSize <= 10000:
    dimPoints = 15
elif datSize < 500000:
    dimPoints = 20
elif datSize < 1000000:
    dimPoints = 25
else:
    dimPoints = 30 # How many eval points for each conditional

f = open(test, 'r')
exec(f.read(), globals())

print('Testing: ', test, '--', testDescript)
print('points per dimension = ', dimPoints)

# For dat file, use the input file name with the .csv extension
gen = gen_data.Gen(test)
data = gen.getDataset(datSize)

prob = ProbSpace(data)
traces = {}
traces['X'] = []
traces['Y'] = []
traces['Z'] = []
v1distr = prob.distr(v1)
v2distr = prob.distr(v2)
v3distr = prob.distr(v3)
v1mean = v1distr.E()
v2mean = v2distr.E()
v3mean = v3distr.E()
fullRange = True
if fullRange:
    v1min = v1distr.percentile(1)
    v1max = v1distr.percentile(99)
    v2min = v2distr.percentile(1)
    v2max = v2distr.percentile(99)
    v3min = v3distr.percentile(1)
    v3max = v3distr.percentile(99)
else:
    v1min = v1distr.percentile(5)
    v1max = v1distr.percentile(95)
    v2min = v2distr.percentile(5)
    v2max = v2distr.percentile(95)
    v3min = v3distr.percentile(5)
    v3max = v3distr.percentile(95)
axMin = min([v1min, v2min, v3min])
axMax = max([v1max, v2max, v3max])
print('axMin, axMax = ', axMin, axMax)

v1dat = prob.ds[v1]
v2dat = prob.ds[v2]
v3dat = prob.ds[v3]
numPts = len(v1dat)
v1Test = np.linspace(v1min, v1max, dimPoints)
v2Test = np.linspace(v2min, v2max, dimPoints)
v3Test = np.linspace(v3min, v3max, dimPoints)


xt = traces['X']
yt = traces['Y']
zt = traces['Z']
my_cmap = plt.get_cmap('gray')

tests = dimPoints**3 # Number of tests

probs = []
testNum = 1
if joint:
    for i in range(1, dimPoints):
        for j in range(1, dimPoints):
            for k in range(1, dimPoints):
                if cumulative:
                    targetSpec = [(v1, None, v1Test[i]), (v2, None, v2Test[j]), (v3, None, v3Test[k])]
                else:
                    targetSpec = [(v1, v1Test[i-1], v1Test[i]), (v2, v2Test[j-1], v2Test[j]), (v3, v3Test[k-1], v3Test[k])]
                p = prob.P(targetSpec)
                if p > 0:
                    print(testNum, '/', tests, ': p = ', p)
                    probs.append(p)
                    xt.append(v2Test[j])
                    yt.append(v3Test[k])
                    zt.append(v1Test[i])
                testNum += 1
else:
    for i in range(1, dimPoints):
        for j in range(1, dimPoints):
            for k in range(1, dimPoints):
                targetSpec = (v1, v1Test[i-1], v1Test[i])
                givensSpec = [(v2, v2Test[j-1], v2Test[j]), (v3, v3Test[k-1], v3Test[k])]
                p = prob.P(targetSpec, givensSpec)
                if p > .001 and p <= 1:
                    print(testNum, '/', tests, ': p = ', p)
                    #probz = prob.P(givensSpec)
                    probs.append(p)
                    xt.append(v2Test[j])
                    yt.append(v3Test[k])
                    zt.append(v1Test[i])
                testNum += 1
pltWidth = 200
pltHeight = 150
fig = plt.figure(constrained_layout=True)
#fig = plt.figure(figsize=(pltWidth, pltHeight))
x = np.array(xt)
y = np.array(yt)
z = np.array(zt)

v1Label = '$' + v1 + '$'
v2Label = '$' + v2 + '$'
v3Label = '$' + v3 + '$'
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel(v2Label, fontsize=20, rotation = 0)
ax.set_ylabel(v3Label, fontsize=20, rotation = 0)
ax.set_zlabel(v1Label, fontsize=20, rotation = 0)
if joint:
    ax.set(title = "P(" + v1 + ", " + v2 + ", " + v3 + ")")
else:
    ax.set(title = "P(" + v1 + " | " + v2 + ", " + v3 + ")")

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
        # First scale to [0, 1]
        prob = prob / median / 2
        #prob = min([1, max([0, prob])])
        
        # Now split high from low
        outProb = (math.tanh((prob-.5) * 2) + 1) / 2
        outProbs.append(outProb)
    return outProbs
scaledProbs = rescale(probs)
maxScaled = max(scaledProbs)
minScaled = min(scaledProbs)
#print('minScaled, maxScaled = ', minScaled, maxScaled, scaledProbs[:1000])

#colors = [my_cmap((1-prob))[:3] + (prob* 1,) for prob in scaledProbs]
#colors = [(.5-prob/4, .5-prob/4, .7-prob/5) + (.2 + prob* .8,) for prob in scaledProbs]
colors = [my_cmap(1-prob) for prob in scaledProbs]
dotsize = np.array(scaledProbs) * (20000 / dimPoints)
dotsize = 2000 / dimPoints
ax.scatter(x, y, z, c=colors, edgecolors='none', marker='o', s=dotsize, linewidth=0)
ax.set_xlim3d(v2min, v2max)
ax.set_ylim3d(v3min, v3max)
ax.set_zlim3d(v1min, v1max)
plt.show()

