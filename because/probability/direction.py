import numpy as np
import math
from because.probability.standardiz import standardize
from sklearn import linear_model
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import svm
from because.probability.rcot.RCoT import RCoT
from sklearn.kernel_ridge import KernelRidge
from because.probability.rff.rffridge import RFFRidgeRegression
from because.probability.rff.rffgpr import RFFGaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier

def balance(A, B):
    newA = []
    newB = []
    bdict = {}
    N = A.shape[0]
    bvals = set(B)
    card = len(bvals)
    if card > 2:
        return (A, B, False)
    return (A, B, True)
    for v in bvals:
        bdict[v] = 0
    for v in B:
        bdict[v] += 1
    proportions = {}
    maxProp = 0
    for v in bvals:
        prop = bdict[v] / N
        if prop > maxProp:
            maxProp = prop
        proportions[v] = bdict[v] / N
    for i in range(N):
        bv = B[i]
        av = A[i]
        for j in range(int(round(maxProp / proportions[bv], 0))):
            newA.append(av)
            newB.append(bv)
    return (np.array(newA), np.array(newB), True)

def test_direction(rvA, rvB, power=1, N_train=2000, sensitivity=None):
    """ When having power parameter less than 1,
        test the causal direction between variables A and B
        using one of the LiNGAM or GeNGAM pairwise algorithms.

        When having power larger than 0, use non-linear method
        to test the causal direction. N_train determines at most
        how many samples would be used to train the non-linear
        model. Currently test uses KNN algorithm.

        Returns a number R.  A positive R indicates that the
        causal path runs from A toward B.  A negative value
        indicates a causal path from B towards A.  Values
        close to zero (e.g. +/- 10**-5) means that causal
        direction could not be determined.
    """
    if power < 1:
        # If power = 0, use lingam (i.e. linear method)
        # Pairwise Lingam Algorithm (Hyperbolic Tangent (HT) variant)
        cum = 0
        s1 = rvA
        s2 = rvB
        for i in range(len(s1)):
            v1 = s1[i]
            v2 = s2[i]
            cumulant = v1 * math.tanh(v2) - v2 * math.tanh(v1)
            cum += cumulant
        avg = cum / float(len(s1))
        cc = np.corrcoef([s1, s2])
        rho = cc[1, 0]
        R = math.tanh(rho * avg * 100)
        return R
    else:
        # We found that averaging multiple small samples (e.g. 2K)
        # is far more accurate and faster than using large or full
        # samples.
        import random
        #newSeed = random.randint(1,1000000)
        #np.random.seed(newSeed)
        sampSize = N_train
        cum = 0.0
        N = len(rvA)
        samples = 3 + int(math.log(power, 10) * 20)
        if N < sampSize * 2:
            sampSize = int(N / 2)
        rvA_a = np.array(rvA)
        rvB_a = np.array(rvB)
        for i in range(samples):
            if sampSize < N:
                inds = np.random.choice(N, size=sampSize, replace=False)
                sA = rvA_a[inds]
                sB = rvB_a[inds]
            else:
                sA = rvA_a
                sB = rvB_a
            #AtoB = non_linear_direct_test(sA, sB)
            #BtoA = non_linear_direct_test(sB, sA)
            try:
            #    pass
                AtoB = non_linear_direct_test(sA, sB)
                BtoA = non_linear_direct_test(sB, sA)
            except:
                continue
            if BtoA == 0 and AtoB == 0:
                continue
            #print('AtoB, BtoA = ', AtoB, BtoA)
            R0 = (BtoA - AtoB) / (BtoA + AtoB)
            Rsamp = math.tanh(R0)
            cum += Rsamp
        R = cum / samples
        #print('AtoB, BtoA, R = ', AtoB, BtoA, R, R0)
        return R

def non_linear_direct_test(A, B):
    A, B, isCat = balance(A,B)
    s1 = A.reshape(-1, 1)
    s2 = B

    N = s1.shape[0]

    #reg = RFFRidgeRegression(rff_dim=100)
    if isCat and False:
        reg = KNeighborsClassifier(n_neighbors=10)
        s2 = np.int_(s2)
    else:
        reg = KNeighborsRegressor(n_neighbors=10)

    reg.fit(s1, s2)

    preds = reg.predict(s1)
    residual = s2 - preds
    #print('N = ', N)
    #for i in range(10):
    #    print('s1, s2, preds = ', s1[i], s2[i], preds[i])

    num_f2 = 8
    #(p, Sta) = RCoT(A, residual, num_f2=num_f2, seed = 1)
    (p, Sta) = RCoT(A, residual, num_f2=num_f2)
    return math.log(Sta / (num_f2 ** 2) + 1)
