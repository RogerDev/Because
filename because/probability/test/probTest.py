""" This is the main test for prob.py.  It uses the data generator:
    Probability/Test/models/probTestDat.py.
    In order to run, you must first generate the test data using
    python3 synth/synthDataGen.py Probability/Test/models/probTestDat.py <numRecords>.
    We typically test with 100,000 records, so that is the recommended
    value for numRecords.
"""
import sys
if '.' not in sys.path:
    sys.path.append('.')
from because.probability import ProbSpace
from because.synth import gen_data
import time

power = 1
dataPoints = 10000
#cMethod = 'j'
cMethod = 'd!'
tries = 10

def run(filename):
    gen = gen_data.Gen(filename)
    dat = gen.getDataset(dataPoints)
    ps = ProbSpace(dat, density=1, power=power, cMethod = cMethod)
    start = time.time()
    print()
    print ('Testing probability module.')
    print()
    print('Testing basic statistics for various types of distribution:')
    print('stats(A) =  ', ps.fieldStats('A'))
    print('stats(C) = ', ps.fieldStats('C'))
    a = ps.distr('A')
    mean = a.mean()
    std = a.stDev()
    print('stats(dice1): mean, std, skew, kurtosis, median, mode = ', mean, std, a.skew(), a.kurtosis(), ' Exp: (3.5, ?, 0, ?)')
    c = ps.distr('C')
    print('stats(d1 + d2): mean, std, skew, kurtosis, median, mode = ', c.E(), c.stDev(), c.skew(), c.kurtosis(), c.median(), c.mode(), ' Exp: (7, ?, 0, ?, 7, 7)')
    d = ps.distr('EXP')
    print('stats(Exponential): mean, std, skew, kurtosis = ', d.E(), d.stDev(), d.skew(), d.kurtosis(), ' Exp: (1, 1, 2, 6)')
    d = ps.distr('IVB')
    print('stats(Logistic): mean, std, skew, kurtosis = ', d.E(), d.stDev(), d.skew(), d.kurtosis(), ' Exp: (0, 1.8138, 0, 1.2)')
    d = ps.distr('N')
    print('stats(Normal):  mean, std, skew, kurtosis, median = ', d.E(), d.stDev(), d.skew(), d.kurtosis(), d.median(), 'Exp: (0, 1, 0, 0)')
    d = ps.distr('N2')
    print('stats(N2: sum of normals):  mean, std, skew, kurtosis = ', d.E(), d.stDev(), d.skew(), d.kurtosis(), 'Exp: (1, 1.414, 0, 0)')
    print()
    print('Testing discrete deterministic probabilities (2-dice -- ala Craps):')
    print('A is Die #1.  B is Die #2.  C is the total of the 2 dice.')
    print('E(B) = ', ps.distr('B').E(), ' Exp: 3.5')
    print('P(B=0) = ', ps.P(('B', 0)), ' Exp: 0')
    print('P(B=1) = ', ps.P(('B', 1)), ' Exp: 1/6 = .166...')
    print('P(B=2) = ', ps.P(('B', 2)), ' Exp: 1/6 = .166...')
    print('P(B >= 0) = ', ps.P(('B', 0, None)), ' Exp: 1.0')
    print('P(B < 0) = ', ps.P(('B', None, 0)), ' Exp: 0.0')
    print('P(-inf <= B > inf) = ', ps.P(('B', None, None)), ' Exp: 1.0')
    print('P(-1 <= B < 3) = ', ps.P(('B', -1, 3)), ' Exp: 1/3')
    print('P(C = 2) =', ps.P(('C', 2)), ' Exp: 1/36 = .0277...')
    print('P(C = 3) =', ps.P(('C', 3)), ' Exp: 1/18 = .055...')
    print('P( 2 <= C < 4) = ', ps.P(('C', 2, 4)), ' Exp: 3/36 = .0833...')
    print('P( 2 <= C < 4 | A = 1) = ', ps.P(('C', 2, 4), ('B', 1)), ' Exp: 1/3')
    print('P( C = 7) = ', ps.P(('C', 7)), ' Exp: 1/6 = .166...')
    print('P( C = 7 | A = 1, B = 6) = ', ps.P(('C', 7), [('A', 1), ('B', 6)]), ' Exp: 1.0')
    print('P( C = 7 | A >= 2, B < 5) = ', ps.P(('C', 7), [('A', 2, None), ('B', None, 5)]), ' Exp: 1/5 = .2')
    print('P(-inf <= A < inf | B >= 1) = ', ps.P(('A', None, None), ('B', 1, None)), ' Exp: 1.0')
    print('P( A >= 3, B >= 3) = ', ps.P([('A', 3, None), ('B', 3, None)]), 'Exp: 4/9 (.444...)')
    print('P( C = 7, A = 5) = ', ps.P([('C', 7), ('A', 5)]), ' Exp: 1/36 (.0277...)')
    print('P( C = 7, A >= 5) = ', ps.P([('C', 7), ('A', 5, None)]), ' Exp: 1/18 (.0555...)')
    print('P( A = 2 | B = 5, C= 7) = ', ps.P(('A', 2), [('B', 5), ('C', 7)]), ' Exp: 1.0')
    print('P( B = 5, C= 7) = ', ps.P(('B', 5), ('C', 7)), ' Exp: 1/6 (.166...)')
    print('P( A = 2, B = 5) = ', ps.P([('A', 2), ('B', 5)]), ' Exp: 1/36 (.0277...)')
    print('P( A = 2, B = 5 | C = 7) = ', ps.P([('A', 2), ('B', 5)], ('C', 7)), ' Exp: 1/6 (.166...)')
    print('P( A = 2, B = 5, N < 0| C = 7) = ', ps.P([('A', 2), ('B', 5), ('N', None, 0)], ('C', 7)), ' Exp: 1/12 (.08333...)')    
    print('E( C | A = 1, B = 6) = ', ps.distr('C', [('A', 1), ('B', 6)]).E(), ' Exp: 7')
    print('E( C | A = 1, B >= 5) = ', ps.distr('C', [('A', 1), ('B', 5, None)]).E(), ' Exp: 6')
    print()
    print('Testing continuous distributions.  Using N = normal(0, 1)')
    n = ps.distr('N')
    mu1 = n.mean()
    mu2 = n.stDev()
    print('stats(N):  mean, std, skew, kurtosis = ', mu1, mu2, n.skew(), n.kurtosis(), 'Exp: (0, 1, 0, 0)')
    print('P( -1 >= N > 1) = ', n.P((-1, 1)), 'Exp: .682')
    print('P( -2 >= N > 2) = ', n.P((-2, 2)), 'Exp: .954')
    print('P( -3 >= N > 3) = ', n.P((-3, 3)), 'Exp: .997')
    print('P( -inf >= N > 0) = ', n.P((None, 0)), 'Exp: .5')
    print('P( 0 >= N > inf) = ', n.P((0, None)), 'Exp: .5')
    print('P( -inf >= N > inf) = ', n.P((None, None)), 'Exp: 1.0')
    print('E( N2 | N = 1) = ', ps.distr('N2', ('N', 1)).E(), ' Exp: 2.0')
    print('E( N2 | 1 <= N < 2) = ', ps.distr('N2',  ('N', 1, 2)).E())
    print()
    print('Dependence testing.  Note: values < .5 are considered independent')
    print('A _||_ B = ', ps.dependence('A', 'B'), ' Exp: < .5')
    print('A _||_ C = ', ps.dependence('A', 'C'), ' Exp: > .5')
    print('B _||_ C = ', ps.dependence('B', 'C'), ' Exp: > .5')
    print('N _||_ N2 = ', ps.dependence('N', 'N2'), ' Exp: > .5')
    print('N _||_ C = ', ps.dependence('N', 'C'), ' Exp: < .5')
    print('C _||_ N = ', ps.dependence('C', 'N'), ' Exp: < .5')
    print('A _||_ B | C >= 8 = ', ps.dependence('A', 'B', [('C', 8, None)]), ' Exp: > .5')
    print('A _||_ B | C < 7 = ', ps.dependence('A', 'B', [('C', None, 7)]), ' Exp: > .5')
    print('A _||_ B | C = 7 = ', ps.dependence('A', 'B', [('C', 7)]), ' Exp: > .5')
    print('A _||_ B | C = 6 = ', ps.dependence('A', 'B', [('C', 6)]), ' Exp: > .5')
    print('A _||_ B | C = 5 = ', ps.dependence('A', 'B', [('C', 5)]), ' Exp: > .5')
    print('A _||_ B | C = 4 = ', ps.dependence('A', 'B', [('C', 4)]), ' Exp: > .5')
    print('A _||_ B | C = 3 = ', ps.dependence('A', 'B', [('C', 3)]), ' Exp: > .5')
    print('A _||_ B | C = 2 = ', ps.dependence('A', 'B', [('C', 2)]), ' Exp: < .5')
    print('A _||_ B | C = 12 = ', ps.dependence('A', 'B', [('C', 12)]), ' Exp: < .5')
    print('A _||_ B | C = ', ps.dependence('A', 'B', ['C']), ' Exp: > .5')
    print()
    print('Independence testing (values > .5 are considered independent):')
    print('A _||_ B = ', ps.independence('A', 'B'), ps.isIndependent('A', 'B'), ' Exp: > .5, True')
    print('A _||_ C = ', ps.independence('A', 'C'), ps.isIndependent('A', 'C'), ' Exp: < .5, False')
    print('A _||_ B | C = ', ps.independence('A', 'B', 'C'), ps.isIndependent('A', 'B', 'C'), ' Exp: < .5, False')
    print('A _||_ N = ', ps.independence('A', 'N'), ps.isIndependent('A', 'N'), ' Exp: > .5, True')
    print()
    print('Testing Conditionalization:')
    ivaDist = ps.distr('IVA')
    ivaMean = ivaDist.E()
    ivaStd = ivaDist.stDev()
    upper = ivaMean + .5*ivaStd
    lower = ivaMean - .5*ivaStd
    diff = upper - lower
    pwr = 2
    print('test interval = ', upper - lower)
    ivcGupper = ps.E('IVC', ('IVA', upper), power=pwr)
    print('E( IVC | IVA = upper)', ivcGupper)
    ivcGlower = ps.E('IVC', ('IVA', lower), power=pwr)
    print('E( IVC | IVA = upper)', ivcGupper)
    print('E( IVC | IVA = lower)', ivcGlower)
    ivcGupper = ps.E('IVC', [('IVA', upper), 'IVB'], power=pwr)
    print('E( IVC | IVA = upper, IVB)', ivcGupper)
    ivcGlower = ps.E('IVC', [('IVA', lower), 'IVB'], power=pwr)
    print('E( IVC | IVA = lower, IVB)', ivcGlower)
    print('ACE(A,C) = ', (ivcGupper - ivcGlower) / diff, ' Exp: ~ 0')
    print()
    print('Testing continuous causal dependence:')
    print('IVB _||_ IVA = ', ps.dependence('IVB', 'IVA'), ' Exp: > .5')
    print('IVA _||_ IVB = ', ps.dependence('IVA', 'IVB'), ' Exp: > .5')
    print('IVB _||_ IVC = ', ps.dependence('IVB', 'IVC'), ' Exp: > .5')
    print('IVA _||_ IVC = ', ps.dependence('IVA', 'IVC'), ' Exp: > .5')
    print('IVA _||_ IVC | IVB = ', ps.dependence('IVA', 'IVC', 'IVB'), ' Exp: < .5')
    print('IVA _||_ IVC | IVB, N = ', ps.dependence('IVA', 'IVC', ['IVB', 'N']), ' Exp: < .5')
    print()
    print('Testing Bayesian Relationships:')
    # P(C=7 | A=5) = P(A=5|C=7) * P(A=5) / P(C=7)
    pA_C = ps.P(('A', 5), ('C', 7))
    pA = ps.P(('A', 5))
    pC = ps.P(('C', 7))
    pC_A = ps.P(('C', 7), ('A', 5))
    invpC_A = pA_C * pA / pC
    err = abs(invpC_A - pC_A)
    print('Inverse P(A=5 | C=7) vs measured (Bayes(P(A | C)), P(A | C), diff): ',invpC_A, pC_A, err, ' Exp: ~ 0' )
    # P(0 <= IVB < 1 | 1 <= IVA < 2) = P(1 <= IVA < 2 | 0 <= IVB < 1) * P(0 <= IVB < 1) / P(1 <= IVA < 2)
    pA_B = ps.P(('IVA', 1, 2), ('IVB', 0, 1))
    pB = ps.P(('IVB', 0, 1))
    pA = ps.P(('IVA', 1, 2))
    pB_A = ps.P(('IVB', 0, 1), ('IVA', 1, 2))
    invpB_A = pA_B * pB / pA
    err = abs(invpB_A - pB_A)
    print('Inverse P(0 <= IVB < 1 | 1 <= IVA < 2) vs measured (Bayes(P(IVB | IVA)), P(IVB | IVA), diff): ',invpB_A, pB_A, err, ' Exp: ~ 0' )
    print()
    print('Testing Prediction and Classification:')
    testDat = {'A':[2, 3, 6], 'B':[5, 2, 6]}
    predDat = ps.Predict('C', testDat)
    for p in range(len(predDat)):
        val = predDat[p]
        a = testDat['A'][p]
        b = testDat['B'][p]
        print('Prediction(C) for A = ', a, ', B = ', b, ', = pred(C) = ', val, ' Exp:', a + b)
    predDat = ps.Classify('C', testDat)
    for p in range(len(predDat)):
        val = predDat[p]
        a = testDat['A'][p]
        b = testDat['B'][p]
        print('Classification(C) for A = ', a, ', B = ', b, ', = pred(C) = ', val, ' Exp:', a + b)    
    testDat = {'N':[.5, 1, 1.5, 2, 2.5, 3], 'B':[1,2,3,4,5,6]}
    predDat = ps.Predict('N2', testDat, cMethod='j')
    for p in range(len(predDat)):
        val = predDat[p]
        n = testDat['N'][p]
        b = testDat['B'][p]
        print('Prediction(N2) for N = ', n, ', B = ', b, ', = pred(C) = ', val, ' Exp:', n + 1)
    predDists = ps.PredictDist('N2', testDat)
    for p in range(len(predDists)):
        d = predDists[p]
        n = testDat['N'][p]
        b = testDat['B'][p]
        print('PredDist(N2) for N = ', n, ', B = ', b, ', = pred(N2 (mean, std)) = ', d.E(), d.stDev(), ' Exp:', n + 1, ', 1')
    print()
    end = time.time()
    duration = end - start
    print('Test Time = ', round(duration))

if __name__ == '__main__':
    if '-h' in sys.argv:
        print('\nMain regression test for prob.py')
        print('\nUsage: python because/probability/test/probTest.py <dataoints>')
        print()
    else:
        filename = "probability/test/models/probTestDat.py"
        if len(sys.argv) > 1:
            dataPoints = int(sys.argv[1])
        run(filename)
