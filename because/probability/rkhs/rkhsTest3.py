import sys
if '.' not in sys.path:
    sys.path.append('.')
import matplotlib.pyplot as plt
from RKHSmod.rkhs import RKHS

FuncAddr =  [[1,2,3,3],
            [1,2,3,4],
            [1,2,3,4.1],
            [1,2,3,4,4.1],
            [1,2,3,4,5,6,7],
            [2,3,4,5],
            [1,2,3,3,3,3,3]]
testPoints = []
testMin = -3
testMax = 10
tp = testMin
numTP = 200
interval = (testMax - testMin) / numTP
# Generate a uniform range of test points.
for i in range(numTP + 1):
    testPoints.append(tp)
    tp += interval
sigma = 1.0
traces = []
for j in range(len(FuncAddr)):
    # Choose a reasonable sigma based on data size.
    r1 = RKHS(FuncAddr[j], kparms=[sigma])
    #r1 = RKHS(X[:size], kparms = [1], k=ksaw)
    fs = []  # The results using a pdf kernel
    totalErr = 0
    deviations = []
    for j in range(len(testPoints)):
        p = testPoints[j]
        fp = r1.F(p)
        fs.append(fp)
    traces.append(fs) # pdf trace
for t in range(len(traces)):
    fs = traces[t]
    label = 'FuncAddr =' + str(FuncAddr[t])
    plt.plot(testPoints, fs, label=label, linestyle='solid')
plt.legend()
plt.show()

