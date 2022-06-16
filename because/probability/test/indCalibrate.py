import sys
if '.' not in sys.path:
    sys.path.append('.')
from because.probability import ProbSpace
from because.synth import read_data
from because.causality import rv
import time

from because.causality import cgraph
from because.probability import independence
from because.probability import prob
from because.probability.standardiz import standardize
import time

#METHOD = 'prob'
#METHOD = 'fcit'
METHOD = 'rcot'
#r1 = 10
num_f = 100
num_f2 = 20
seed = None
POWER = 1
print('power = ', POWER)
args = sys.argv
test = 'C:/Users/91952/Because/because/probability/test/models/indCalibrationDat.csv'

r = read_data.Reader(test)
dat = r.read()
vars = dat.keys()
for var in vars:
    dat[var] = standardize(dat[var])
#print('dat = ', dat)
ps = ProbSpace(dat, power = POWER)

# List a variety of independent relationships
indeps = [  # totally independence
            ('L1', 'L2'),
            ('L1', 'L2', ['A']),
            ('L1', 'L2', ['A', 'B']),
            ('L2', 'L3'),
            ('L1', 'L3'),
            ('E1', 'E2'),
            ('N1', 'N2'),
            ('L4', 'L5'),
            ('L5', 'L6'),
            ('L4', 'N3'),
            ('C', 'E2'),
            # common cause
            ('L6', 'L7', ['L3']),
            ('L4', 'L6', ['L3']),
            ('L8', 'L9', ['L1']),
            # complex causal and v-structure
            ('B', 'D', ['A']),
            ('A', 'C', ['B', 'D']),
            # v-structure and common cause
            ('M1', 'E2'),
            ('M1', 'E2'),
            ]

# List a varieety of dependent relationships
deps = [
        ('L3', 'L4'),
        ('L5', 'L2'),
        ('L6', 'L3'),
        ('L6', 'L7'),
        ('L7', 'L4'),
        ('E3', 'E1'),
        ('E3', 'E2'),
        ('M1', 'N2'),
        ('B', 'D'),
        ('B', 'D', 'C'),
        ('B', 'D', ['A', 'C']),
        ('B', 'A', 'C'),
        ('B', 'A', ['C', 'D']),
        ('B', 'C', 'A'),
        ('A', 'C', 'B'),
        ('L8', 'L9'),
        ('N1', 'N2', ['N3']),
        ('N3', 'E1', ['M1']),
        ('M1', 'E2', ['E3'])
        ]
print('Testing: ', test)
start = time.time()

testVal = 0
condTestVal = 0
delta = .1

minIndep = 999999.0
maxDep = 0.0
minDep = 9999999.0
cumIndep = 0.0
cumDep = 0.0
print()
print('Testing expected independents:')
for ind in indeps:
    start2 = time.time()
    if len(ind) == 2:
        x, y = ind
        z = []
    elif len(ind) == 3:
        x, y, z = ind
    else:
        print('*** Error, improperly specified independence =', ind)
    #xD = [dat[x]]
    #yD = [dat[y]]
    #zD = [dat[zvar] for zvar in z]
    pval = independence.test(ps, [x], [y], z, method = METHOD, power = POWER, seed=seed, num_f=num_f, num_f2=num_f2)
    #pval = ps.independence(x, y, z)
    if pval < minIndep:
        minIndep = pval
    print('dependence', ind, '= ', 1-pval)
    end = time.time()
    duration = end - start2
    print('Test Time = ', round(duration))

print()
print('Testing expected dependents:')
print()
for dep in deps:
    start2 = time.time()
    if len(dep) == 2:
        x, y = dep
        z = []
    elif len(dep) == 3:
        x, y, z = dep
    else:
        print('*** Error, improperly specified independence =', dep)
    #xD = [dat[x]]
    #yD = [dat[y]]
    #zD = [dat[zvar] for zvar in z]
    pval = independence.test(ps, [x], [y], z, method = METHOD, power = POWER, seed=seed, num_f=num_f, num_f2=num_f2)
    #pval = ps.independence(x, y, z)
    if pval > maxDep:
        maxDep = pval
    if pval < minDep:
        minDep = pval
    print('dependence', dep, ' = ', 1-pval)
    end = time.time()
    duration = end - start2
    print('Test Time = ', round(duration))
print()

print('Maximum dependence for expected independents = ', 1-minIndep)
print('Minimum dependence for expected dependents =', 1- maxDep)
print('Margin = ', minIndep - maxDep, '.  Positive margin is good.')
print('Maximum dependence = ', 1-minDep)
print('best Low threshold is: ', max([((1-minIndep) + (1- maxDep)) / 2.0, 1-minIndep + .001]))
print('best High threshold is: ', 1 - minDep + .1)
print()
end = time.time()
duration = end - start
print('Test Time = ', round(duration))