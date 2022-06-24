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

alpha = .01

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
f = open(test, 'r')
exec(f.read(), globals())

print('Testing: ', test, '--', testDescript)

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
#ymin = targdistr.minVal()
#ymax = targdistr.maxVal()
#xmin = conddistr.minVal()
#xmax = conddistr.maxVal()
lim = 3
v1dat = prob.ds[v1]
v2dat = prob.ds[v2]
v3dat = prob.ds[v3]
numPts = len(v1dat)
xt = traces['X']
yt = traces['Y']
zt = traces['Z']
for i in range(numPts):
    #yval = ymin + i * y_incr        
    xt.append(v1dat[i])
    yt.append(v2dat[i])
    zt.append(v3dat[i])
fig = plt.figure(figsize=(200, 150))
x = np.array(xt)
y = np.array(yt)
z = np.array(zt)
v1Label = '$' + v1 + '$'
v2Label = '$' + v2 + '$'
v3Label = '$' + v3 + '$'
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel(v1Label, fontsize=20, rotation = 0)
ax.set_ylabel(v2Label, fontsize=20, rotation = 0)
ax.set_zlabel(v3Label, fontsize=20, rotation = 0)
#ax.plot_trisurf(x, y, z, cmap = my_cmap)
ax.scatter(x, y, z, color='blue', linewidth=5, alpha=alpha);
plt.show()
