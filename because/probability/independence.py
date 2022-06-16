from because.probability import prob
from because.probability.standardiz import standardize

# Tests for independence between two sets of variables,
# optionally given a third set.
# X, Y, and Z are each a list of data series', one series
# per variable, with each series representing the values at
# each sample.
# For example, Z with 2 variables and N samples would be:
# [[v1[0], ... v1[N-1]], [v2[1], ... , v2[N-1]]]
# Returns a p-value for the null hypothesis that:
# X and Y are Dependent given Z.  P-values less than
# .05 provide a 95% confidence that the variables are dependent.
# P-values > .05 imply independence (i.e. lack of proof of dependence).
def testFCIT(ps, x, y, z=[]):
    import numpy as np
    from fcit import fcit
    X = [ps.ds[x[0]]]
    Y = [ps.ds[y[0]]]
    Z = []
    for var in z:
        zdat = ps.ds[var]
        Z.append(zdat)
    Xa = np.array(X).transpose()
    #print('xshape = ', Xa.shape)
    Ya = np.array(Y).transpose()
    if Z:
        Za = np.array(Z).transpose()
        #print('zshape = ', Za.shape)
        pval = fcit.test(Xa, Ya, Za, num_perm = 10, prop_test = .40)
    else:
        pval = fcit.test(Xa, Ya, num_perm = 10, prop_test = .40)
    return pval

def testSDCIT(ps, x, y, z=[]):
    import numpy as np
    from sdcit.sdcit_mod import SDCIT
    from sdcit.utils import rbf_kernel_median
    X = [ps.ds[x[0]]]
    Y = [ps.ds[y[0]]]
    Z = []
    for var in z:
        zdat = ps.ds[var]
        Z.append(zdat)
    Xa = np.array(X).transpose()
    Ya = np.array(Y).transpose()
    if not Z:
        return testFCIT(ps, X, Y)
    Za = np.array(Z).transpose()
    Kx, Ky, Kz = rbf_kernel_median(Xa, Ya, Za)
    test_stat, p_value = SDCIT(Kx, Ky, Kz)
    #print('p = ', p_value)
    return p_value

def testProb(ps, X, Y, Z=[], power=2):
    X = X[0]
    Y = Y[0]
    #print('ps, X, Y, Z = ', ps, X, Y, Z)
    ind = ps.independence(X, Y, Z, power=power)
    return ind

def testRCoT(ps, X, Y, Z=[], seed=None, num_f=100, num_f2=5):
    X = X[0]
    Y = Y[0]
    #print('ps, X, Y, Z = ', ps, X, Y, Z)
    ind = ps.independence(X, Y, Z, dMethod='rcot', seed=seed, num_f=num_f, num_f2=num_f2)
    return ind

def test(ps, X, Y, Z=[], method=None, power=1, seed=None, num_f=100, num_f2=5):
    # Valid values for method are: None(default), 'prob', 'fcit', 'sdcit'
    if method is None:
        method = 'prob'
    if method == 'fcit':
        p_val = testFCIT(ps, X, Y, Z)
    elif method == 'sdcit':
        p_val = testSDCIT(ps, X, Y, Z)
    elif method == 'prob':
        p_val = testProb(ps, X, Y, Z, power = power)
    elif method == 'rcot':
        p_val = testRCoT(ps, X, Y, Z, seed=seed, num_f=num_f, num_f2=num_f2)
    else:
        print('independence.test:  method = ', method, 'is not supported.')
    return p_val