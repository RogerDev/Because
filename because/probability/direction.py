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

def test_direction(rvA, rvB, power=1, N_train=100000):
    """
    """
    if power <= 1:
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
        R = rho * avg
        return R
    else:
        AtoB = non_linear_direct_test(rvA, rvB, N_train)
        BtoA = non_linear_direct_test(rvB, rvA, N_train)
        R = AtoB - BtoA
        return R/1000

def non_linear_direct_test(A, B, N_train=100000):

    s1 = np.array(A).reshape(-1, 1)
    s2 = np.array(B)

    N = s1.shape[0]

    if N_train < N:
        inds = np.random.choice(N, size=N_train, replace=False)
        s1 = s1[inds]
        s2 = s2[inds]
        A = np.array(A)[inds]

    #reg = RFFRidgeRegression(rff_dim=100)

    reg = KNeighborsRegressor(n_neighbors=10)

    reg.fit(s1, s2)

    residual = s2 - reg.predict(s1)
    (p, Sta) = RCoT(A, residual)
    return p[0]

'''
def test_direction(rvA, rvB):
    """ Test the causal direction between variables A and B
        using one of the LiNGAM or GeNGAM pairwise algorithms.
        Returns a number R.  A positive R indicates that the
        causal path runs from A toward B.  A negative value
        indicates a causal path from B towards A.  Values
        close to zero (e.g. +/- 10**-5) means that causal
        direction could not be determined.
    """
    s1 = standardize(rvA)
    s2 = standardize(rvB)
    # Pairwise Lingam Algorithm (Hyperbolic Tangent (HT) variant)
    cum = 0
    for i in range(len(s1)):
        v1 = s1[i]
        v2 = s2[i]
        cumulant = v1*math.tanh(v2) - v2*math.tanh(v1)
        cum += cumulant
    avg = cum / float(len(s1))
    cc = np.corrcoef([s1,s2])
    rho = cc[1,0]
    R = rho * avg
    return R
'''
