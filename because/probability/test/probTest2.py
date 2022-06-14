
"""
Temporary test for assessing various conditionalizing and independence testing algorithms.
"""
import sys
if '.' not in sys.path:
    sys.path.append('.')
from because.probability import ProbSpace
from because.synth import read_data
from because.synth import gen_data
import time

power = 2
dataPoints = 10000
#cMethod = 'j'
#cMethod = 'd!'
cMethod = 'd!'
tries = 1
smoothness=.25

def run(filename):
    gen = gen_data.Gen(filename)
    dat = gen.getDataset(dataPoints)
    ps = ProbSpace(dat, density=1, power=power, cMethod = cMethod)
    start = time.time()
    print()
    print ('Testing probability module.')

    def condit():
        print('Testing Conditionalization:')
        cumAce = 0.0
        maxAce = -9999999
        minAce = 9999999
        for i in range(tries):
            dat = gen.getDataset(dataPoints)
            ps = ProbSpace(dat, density=1, power=power, cMethod = cMethod)
            print('Std Devs:  IVA, IVB, IVC = ', ps.distr('IVA').stDev(), ps.distr('IVB').stDev(), ps.distr('IVC').stDev())
            print('Variances:  IVA, IVB, IVC = ', ps.distr('IVA').stDev()**2, ps.distr('IVB').stDev()**2, ps.distr('IVC').stDev()**2)
            ivaDist = ps.distr('IVA')
            ivaMean = ivaDist.E()
            ivaStd = ivaDist.stDev()
            upper = ivaMean + .5
            lower = ivaMean - .5
            diff = upper - lower
            print('test interval = ', lower, upper, upper - lower)
            ivcGupper = ps.E('IVC', ('IVA', upper))
            print('E( IVC | IVA = upper)', ivcGupper)
            ivcGlower = ps.E('IVC', ('IVA', lower))
            print('E( IVC | IVA = lower)', ivcGlower)
            ivcGupper = ps.E('IVC', [('IVA', upper), 'IVB'], smoothness=smoothness)
            print('E( IVC | IVA = upper, IVB)', ivcGupper)
            ivcGlower = ps.E('IVC', [('IVA', lower), 'IVB'], smoothness=smoothness)
            print('E( IVC | IVA = lower, IVB)', ivcGlower)
            ace = (ivcGupper - ivcGlower) / diff
            print('ACE(A,C) = ', ace, ' Exp: ~ 0')
            cumAce += ace
            minAce = min([minAce, ace])
            maxAce = max([maxAce, ace])
            print()
        print()
        print('ACE(avg, min, max, range) = ', cumAce/tries, minAce, maxAce, maxAce - minAce)
        print()

    def depend():
        print('Testing continuous causal dependence:')
        print('IVB _||_ IVA = ', ps.dependence('IVB', 'IVA'), ' Exp: > .5')
        print('\n')
        print('IVA _||_ IVB = ', ps.dependence('IVA', 'IVB'), ' Exp: > .5')
        print('\n')
        print('IVB _||_ IVC = ', ps.dependence('IVB', 'IVC'), ' Exp: > .5')
        print('\n')
        print('IVA _||_ IVC = ', ps.dependence('IVA', 'IVC'), ' Exp: > .5')
        print('\n')
        print('IVA _||_ IVC | IVB = ', ps.dependence('IVA', 'IVC', 'IVB'), ' Exp: < .5')
        print('\n')
        print('IVA _||_ IVC | IVB, N = ', ps.dependence('IVA', 'IVC', ['IVB', 'N']), ' Exp: < .5')
        print('\n')
        print()
        print('Testing Bayesian Relationships:')

    condit()
    depend()
    print()
    end = time.time()
    duration = end - start
    print('Test Time = ', round(duration))

if __name__ == '__main__':
    if '-h' in sys.argv:
        print('\nMain regression test for prob.py')
        print('\nUsage: python because/probability/test/probTest.py')
        print()
    else:
        if len(sys.argv) >= 2:
            dataPoints = int(sys.argv[1])
        filename = "probability/test/models/probTestDat.py"
        run(filename)
