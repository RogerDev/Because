import numpy as np
import math
from because.probability.standardiz import standardize

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

