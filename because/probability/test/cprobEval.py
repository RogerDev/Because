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

    test = 'models/nCondition.py'

    f = open(test, 'r')
    exec(f.read(), globals())
    if not quiet:
        print('Testing: ', test, '--', testDescript)

    # For dat file, use the input file name with the .csv extension
    tokens = test.split('.')
    testFileRoot = str.join('.',tokens[:-1])
    datFileName = testFileRoot + '.csv'
    dp_results = []
    jp_results = []
    up_results = []
    dp_run = []
    jp_run = []
    up_run = []
    for i in range(tries):
        if not quiet:
            print('\nRun', i+1)
        gen = gen_data.Gen(test)
        sdg = gen.generate(datSize)
        d = read_data.Reader(datFileName, quiet=quiet)
        data = d.read()
        prob1 = ProbSpace(data, cMethod = 'd!') # D-Prob
        prob2 = ProbSpace(data, cMethod = 'j') # J-Prob
        prob3 = ProbSpace(data, cMethod = 'u') # U-Prob

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
        # Generate the target values for comparison
        for t in tps:
            cmpr1 = tanh(t[0]+1)
            cmpr2 = sin(t[1]*.75) if dims > 2 else 0
            cmpr3 = tanh(t[2]-2) if dims > 3 else 0
            #print('t[3] = ', t[3])
            cmpr4 = cos(t[3]*1.2) if dims > 4 else 0
            cmpr5 = tanh(t[4]+3) if dims > 5 else 0
            cmprL = [cmpr1, cmpr2, cmpr3, cmpr4, cmpr5]
            cmpr = sum(cmprL[:dims - 1])
            cmprs.append(cmpr)
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
        jp_start = time.time()
        tnum = 0
        for t in tps:
            tnum += 1
            try:
                condspec = []
                for c in range(dims-1):
                    condVar = cond[c]
                    val = t[c]
                    spec = (condVar, val)
                    condspec.append(spec)
                y_x = prob2.E(target, condspec)
            except:
                print('got exception -- JP')
                y_x = 0
            jp_est.append(y_x)
        prob2 = None
        jp_end = time.time()
        up_start = time.time()
        for t in tps:
            try:
                condspec = []
                for c in range(dims-1):
                    condVar = cond[c]
                    val = t[c]
                    spec = (condVar, val)
                    condspec.append(spec)
                y_x = prob3.E(target, condspec)
            except:
                print('got exception -- UP')
                y_x = 0
            up_est.append(y_x)
        prob3 = None
        up_end = time.time()
        totalErr_dp = 0.0
        totalErr_jp = 0.0
        totalErr_up = 0.0
        results = []
        ysum = 0.0
        for i in range(len(cmprs)):
            t = tps[i]
            cmpr = cmprs[i]
            ysum += cmpr
            dp_e = dp_est[i]
            jp_e = jp_est[i]
            up_e = up_est[i]
            error2_dp = (cmpr-dp_e)**2
            error2_jp = (cmpr-jp_e)**2
            error2_up = (cmpr-up_e)**2
            totalErr_dp += error2_dp
            totalErr_jp += error2_jp
            totalErr_up += error2_up
            results.append((t, dp_e, jp_e, up_e, cmpr, error2_dp, error2_jp, error2_up))
        rmse_dp = sqrt(totalErr_dp) / len(tps)
        rmse_jp = sqrt(totalErr_jp) / len(tps)
        rmse_up = sqrt(totalErr_up) / len(tps)
        # Calc R2 for each
        yavg = ysum / len(tps)
        ssTot = sum([(c - yavg)**2 for c in cmprs])
        R2_dp = max([1 - totalErr_dp / ssTot, 0.0])
        R2_jp = max([1 - totalErr_jp / ssTot, 0.0])
        R2_up = max([1 - totalErr_up / ssTot, 0.0])
        if not quiet:
            print('DP:')
            print('   R2 =', R2_dp)
            print('JP:')
            print('   R2 =', R2_jp)
            print('UP:')
            print('   R2 =', R2_up)
        dp_runtime = round(dp_end - dp_start, 5)
        jp_runtime = round((jp_end - jp_start), 5)
        up_runtime = round(up_end - up_start, 5)
        dp_results.append(R2_dp)
        jp_results.append(R2_jp)
        up_results.append(R2_up)
        dp_run.append(dp_runtime)
        jp_run.append(jp_runtime)
        up_run.append(up_runtime)
    dp_avg = round(np.mean(dp_results),3)
    jp_avg = round(np.mean(jp_results),3)
    up_avg = round(np.mean(up_results),3)
    dp_min = round(np.min(dp_results),3)
    jp_min = round(np.min(jp_results),3)
    up_min = round(np.min(up_results),3)
    dp_max = round(np.max(dp_results),3)
    jp_max = round(np.max(jp_results),3)
    up_max = round(np.max(up_results),3)
    dp_std = round(np.std(dp_results),3)
    jp_std = round(np.std(jp_results),3)
    up_std = round(np.std(up_results),3)
    dp_runt = round(np.mean(dp_run),3)
    jp_runt = round(np.mean(jp_run),3)
    up_runt = round(np.mean(up_run),3)
    print('dims, datSize, tries = ', dims, datSize, tries)
    print('Average R2: DP, JP, UP = ', dp_avg, jp_avg, up_avg)
    print('Min R2: DP, JP, UP = ', dp_min, jp_min, up_min)
    print('Max R2: DP, JP, UP = ', dp_max, jp_max, up_max)
    print('Std R2: DP, JP, UP = ', dp_std, jp_std, up_std)
    print('Runtimes: DP, JP, UP = ', dp_runt, jp_runt, up_runt)
    print('NumTests = ', tries)
    print('Uprob: tau = ', uprob.tau, ', minTau = ', uprob.mintau)
    lmbda, Dfilter, Ntarg = uprob.calcParms(None, datSize, dims)

    print('Uprob: Lambda = ', lmbda)
    print('Uprob: DFilter =', Dfilter, 'Ntarg =', Ntarg)
    print('*************************************', flush=True)
    return (datSize, dims, tries, dp_avg, jp_avg, up_avg, dp_runt, jp_runt, up_runt)

if __name__ == '__main__':
    tries = 10
    datSize = 1000
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
            dims = 2
        if len(sys.argv) > 2:
            datSize = int(sys.argv[2])
        if len(sys.argv) > 3:
            tries = int(sys.argv[3])
        if len(sys.argv) > 4:
            condPoints = int(sys.argv[4])
        if len(sys.argv) > 5:
            lim = int(sys.argv[5])
        run(dims, datSize, tries, lim)