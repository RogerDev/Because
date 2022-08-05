"""
Compares different probability methods using ProbSpace with different
cMethod parameters.
Compares D-Prob, J-Prob and U-Prob using the models/nCondition.py SEM file.
Can run the test multiple times and average the results, which is useful
for small data sizes, where random variation can be large.
Tests from 1 to 5 conditionals (by setting dims parameter between 2 and 6)
Command parameters allow adjustment of data size, dims, and number of tries
to average over.
"""
import sys
if '.' not in sys.path:
    sys.path.append('.')
import time
from math import log, tanh, sqrt, sin, cos

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
from numpy.random import *

from because.causality import rv
from because.synth import read_data, gen_data
from because.probability import independence
from because.probability import ProbSpace
from because.probability.rkhs.rkhsMV import RKHS
from because.probability import uprob

def run(dims, datSize, tries=1, lim=3, quiet=False):

    # Arg format is <dims> <datSize> <tries>

    #print('dims, datSize, tries, lim = ', dims, datSize, tries, lim)
    numTests = 200
    if not quiet:
        print('Test Limit = ', lim, 'standard deviations from mean')
        print('Dimensions = ', dims, '.  Conditionals = ', dims - 1)
        print('Number of points to evaluate = ', numTests)

    test = 'models/nCondition2.py'

    f = open(test, 'r')
    exec(f.read(), globals())
    if not quiet:
        print('Testing: ', test, '--', testDescript)

    # For dat file, use the input file name with the .csv extension
    tokens = test.split('.')
    testFileRoot = str.join('.',tokens[:-1])
    datFileName = testFileRoot + '.csv'
    dp_results = []
    ml_results = []
    dp_run = []
    ml_run = []
    for i in range(tries):
        if not quiet:
            print('\nRun', i+1)
        gen = gen_data.Gen(test)
        sdg = gen.generate(datSize)
        d = read_data.Reader(datFileName, quiet=quiet)
        data = d.read()
        prob1 = ProbSpace(data, categorical=['A2', 'A3', 'A4', 'A5', 'A6'], cMethod = 'd') # D-Prob
        prob4 = ProbSpace(data, categorical=['A2', 'A3', 'A4', 'A5', 'A6'], cMethod = 'ml')  # ML-Prob

        N = prob1.N
        vars = prob1.fieldList
        cond = []
        # Get the conditional variables
        for i in range(len(vars)):
            var = vars[i]
            if var[0] != 'A':
                cond.append(var)
        # There is a target: 'A<dims>' for each conditional dimension.  So for 3D (2 conditionals),
        # we would use A3.
        target = 'A' + str(dims)

        smoothness=1.0
        evaluations = 0
        results = []
        totalErr_dp = 0
        totalErr_jp = 0
        totalErr_up = 0
        totalErr_ml = 0
        conds = len(cond)
        tps = []
        evaluations = 0
        means = [prob1.E(c) for c in cond]
        stds = [prob1.distr(c).stDev() for c in cond]
        minvs = [means[i] - stds[i] * lim for i in range(len(means))]
        maxvs = [means[i] + stds[i] * lim for i in range(len(means))]

        # Generate the test points
        for i in range(numTests):
            tp = []
            for j in range(dims-1):
                v = uniform(minvs[j], maxvs[j])
                tp.append(v)
            tps.append(tp)      

        tnum = 0
        ssTot = 0 # Total sum of squares for R2 computation
        cmprs = []
        dp_est = []
        jp_est = []
        up_est = []
        ml_est = []
        # Generate the target values for comparison
        for t in tps:
            cmpr1 = 5*tanh(t[0]+1)
            cmpr2 = sin(t[1]*.75) if dims > 2 else 0
            cmpr3 = tanh(t[2]-2) if dims > 3 else 0
            #print('t[3] = ', t[3])
            cmpr4 = cos(t[3]*1.2) if dims > 4 else 0
            cmpr5 = tanh(t[4]+3) if dims > 5 else 0
            cmprL = [cmpr1, cmpr2, cmpr3, cmpr4, cmpr5]
            cmpr = sum(cmprL[:dims - 1])
            label = 0 if cmpr < -0.5 else 1 if -0.5 <= cmpr < 0 else 2 if 0 <= cmpr < 0.5 else 3
            cmprs.append(label)
        dp_start = time.time()
        for t in tps:
            try:
                condspec = []
                for c in range(dims-1):
                    condVar = cond[c]
                    val = t[c]
                    spec = (condVar, val)
                    condspec.append(spec)
                y_x = prob1.E(target, condspec)
            except:
                print('got exception -- DP')
                y_x = 0
            dp_est.append(y_x)
        prob1 = None
        dp_end = time.time()
        ml_start = time.time()
        for t in tps:
            try:
                condspec = []
                for c in range(dims - 1):
                    condVar = cond[c]
                    val = t[c]
                    spec = (condVar, val)
                    condspec.append(spec)
                y_x = prob4.E(target, condspec)
            except:
                print('got exception -- ML')
                y_x = 0
            ml_est.append(y_x)
        prob4 = None
        ml_end = time.time()
        correct_dp = 0.0
        correct_ml = 0.0
        results = []
        for i in range(len(cmprs)):
            t = tps[i]
            cmpr = cmprs[i]
            dp_e = dp_est[i]
            ml_e = ml_est[i]
            correct_dp += cmpr == dp_e
            correct_ml += cmpr == ml_e
            results.append((t, dp_e, ml_e, cmpr, cmpr == dp_e, cmpr == ml_e))

        Acc_dp = correct_dp / numTests
        Acc_ml = correct_ml / numTests
        if not quiet:
            print('DP:')
            print('   Acc =', Acc_dp)
            print('ML:')
            print('   Acc =', Acc_ml)
        dp_runtime = round(dp_end - dp_start, 5)
        ml_runtime = round(ml_end - ml_start, 5)
        dp_results.append(Acc_dp)
        ml_results.append(Acc_ml)
        dp_run.append(dp_runtime)
        ml_run.append(ml_runtime)
    dp_avg = round(np.mean(dp_results),3)
    ml_avg = round(np.mean(ml_results), 3)
    dp_min = round(np.min(dp_results),3)
    ml_min = round(np.min(ml_results), 3)
    dp_max = round(np.max(dp_results),3)
    ml_max = round(np.max(ml_results), 3)
    dp_std = round(np.std(dp_results),3)
    ml_std = round(np.std(ml_results), 3)
    dp_runt = round(np.mean(dp_run),3)
    ml_runt = round(np.mean(ml_run), 3)
    print('dims, datSize, tries = ', dims, datSize, tries)
    print('Average R2: DP, JP, UP, ML = ', dp_avg, ml_avg)
    print('Min R2: DP, JP, UP, ML = ', dp_min, ml_min)
    print('Max R2: DP, JP, UP, ML = ', dp_max, ml_max)
    print('Std R2: DP, JP, UP, ML = ', dp_std, ml_std)
    print('Runtimes: DP, JP, UP, ML = ', dp_runt, ml_runt)
    print('NumTests = ', tries)
    print('Uprob: tau = ', uprob.tau, ', minTau = ', uprob.mintau)
    lmbda, Dfilter, Ntarg = uprob.calcParms(None, datSize, dims)

    print('Uprob: Lambda = ', lmbda)
    print('Uprob: DFilter =', Dfilter, 'Ntarg =', Ntarg)
    print('*************************************', flush=True)
    return (datSize, dims, tries, dp_avg, ml_avg, dp_runt, ml_runt)

if __name__ == '__main__':
    tries = 10
    datSize = 10000
    lim = 3
    if '-h' in sys.argv:
        print('\nUsage: python because/probability/test/cprobEval.py [dims] [datSize] [tries]')
        print('   dims is the number of dimensions to test with (2-6).  Default 2.')
        print('     dims is the number of conditionals plus 1.  E.g. dims=6 uses 5 conditionals.')
        print('   tries is the number of times to repeat the experiment and average the results.')
        print('     Note that each try is based on a different dataset sample.')
        print('   datSize is the number of data records to generate.')
    else:
        if len(sys.argv) > 1:
            dims = int(sys.argv[1])
        else:
            dims = 6
        if len(sys.argv) > 2:
            datSize = int(sys.argv[2])
        if len(sys.argv) > 3:
            tries = int(sys.argv[3])
        if len(sys.argv) > 4:
            condPoints = int(sys.argv[4])
        if len(sys.argv) > 5:
            lim = int(sys.argv[5])
        run(dims, datSize, tries, lim)