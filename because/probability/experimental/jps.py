"""
Compute the number of potential qualitatively different probability queries given the number of variables. 
"""
import math
fac = math.factorial


def C(n, k):
    """ 
    Combinations
    """
    return fac(n) / (fac(k) * fac(n-k))

def numQueries(d):
    """
    Return the number of potential queries given the dimension of the dataset.
    """
    jointQ = 0
    for j in range(1,d):
        jointQ += C(d, j)
    condQ = 0
    for p in range(1, d):
        for q in range(1, d-p+1):
            condQ += C(d,p) * C(d-p, q)
    return 1 + jointQ + condQ

if __name__ == __main__:
    print(numQueries(3))
    print(numQueries(12))
    print(numQueries(32))