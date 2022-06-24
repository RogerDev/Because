import numpy as np
import sys
import math
from .pgamma import pgamma
from .hbe import hbe
from .utils import *
#import icecream as ic

def vapply(X,fun,v):
    res = []
    for i in range(X.shape[0]):
        # print("X[i]: ",X[i])
        res.append(fun(int(X[i]),v))
    res = np.array(res)
    return res

def mapply(fun,ivec,jvec,vec1,vec2):
    res=[]
    for i in ivec:
        l = []
        for j in jvec:
            l.append(fun(int(i),int(j),vec1,vec2))
        res.append(l)
    return res

def rep(x,y):
    return np.repeat(x,y)

def choose(n,kvec):
    ans = []
    for k in kvec:
        try:
            comb = math.comb(n, k)
        except AttributeError:
            comb = int(math.factorial(n) / (math.factorial(k) * math.factorial(n-k))) if n >= k else 0
        ans.append(comb)
    return np.array(ans)

def c(x, vec):
    return np.insert(vec,0,x)

def polyroot(vec):
    # FIXME: I think we have to reverse the vector 
    # as per the def in R and python
    return np.polynomial.polynomial.polyroots(vec)

def Re(vec):
    res = [n for n in vec if np.isreal(n)]
    return res

# def uniroot(fun, limit, m_vec, N, tol):
#     from scipy.optimize import root
#     x = root(fun, limit[0], args=(m_vec, N), tol=tol)
#     print("x.x: ",x.x)
#     return x.x[0]

def uniroot(fun, limit, m_vec, N, tol):
    from scipy.optimize import brentq as root
    x = root(fun,limit[0],limit[1], args=(m_vec, N), xtol=tol)
    return x

def factorial(vec):
    ans = []
    for x in vec:
        ans.append(np.math.factorial(x))
    return np.array(ans)

def lpb4(coeff, x):
    if(coeff.any() == None or x == None):
        print("missing an argument - need to specify \"coeff\" and \"x\"")
        return None

    # TODO: CHECK if it correct
    if (checkCoeffsArePositiveError(coeff)):
        exit(getCoeffError(coeff))

    if (checkXvaluesArePositiveError([x])):
        exit(getXvaluesError(x))

    if(len(coeff) < 4):
        print(
            "Less than four coefficients - LPB4 method may return NaN: running hbe instead.")
        # TODO:
        return(hbe(coeff, x))

    # step 0: decide on parameters for distribution and support points p specified to be 4 for this version of the function
    p = 4

    # step 1: Determine/compute the moments m_1(H), ... m_2p(H)
    moment_vec = get_weighted_sum_of_chi_squared_moments(coeff, p)
    # print("moment_vec: ", moment_vec)

    lambdatilde_1 = get_lambdatilde_1(moment_vec[0], moment_vec[1])
    # print("lambdatilde_1 ",lambdatilde_1 )

    bisect_tol = 1e-6
    lambdatilde_p = get_lambdatilde_p(lambdatilde_1, p, moment_vec, bisect_tol)
    # print("lambdatilde_p ",lambdatilde_p )

    M_p = deltaNmat_applied(lambdatilde_p, moment_vec, p)
    # print("M_p ",M_p )

    mu_poly_coeff_vec = get_Stilde_polynomial_coefficients(np.array(M_p))
    # print("mu_poly_coeff_vec",mu_poly_coeff_vec)

    mu_roots = get_real_poly_roots(mu_poly_coeff_vec)

    pi_vec = gen_and_solve_VDM_system(M_p, mu_roots)

    mixed_p_val_vec = get_mixed_p_val_vec(x, mu_roots, pi_vec, lambdatilde_p)

    return mixed_p_val_vec
 

def get_weighted_sum_of_chi_squared_moments(coeffvec, p):
    # Checked - giving correct
    cumulant_vec = get_cumulant_vec_vectorised(coeffvec, p)
    moment_vec = get_moments_from_cumulants(cumulant_vec)
    return (moment_vec)

def get_cumulant_vec_vectorised(coeffvec, p):
    index = np.arange(1,(2*p)+1)

    #TODO: vapply
    cumulant_vec = (2**(index-1)) * factorial(index-1) * vapply(index, sum_of_powers, coeffvec)

    return cumulant_vec

def sum_of_powers(index, v):
    return np.sum(v**index)

def get_moments_from_cumulants(cumulant_vec):
    moment_vec = np.copy(cumulant_vec)
    if(len(moment_vec) > 1):
        # FIXME: linspace
        for n in range(1,len(moment_vec)):
            moment_vec[n] = moment_vec[n] + update_moment_from_lower_moments_and_cumulants(n+1, moment_vec, cumulant_vec)
    return moment_vec

def update_moment_from_lower_moments_and_cumulants(n, moment_vec, cumulant_vec):
    # m <- c(1:(n-1))
    m = np.arange(1,n)
    #TODO: choose
    sum_of_additional_terms = np.sum(choose(n-1, m-1) * cumulant_vec[m-1] * moment_vec[n-m-1])
    return sum_of_additional_terms

def get_lambdatilde_1(m1, m2):
    return (m2/(m1**2)-1)

def deltaNmat_applied(x, m_vec, N):
    '''Compute the delta_N matrix
    '''
    Nplus1 = N+1
    # moments 0, 1, ..., 2N
    m_vec = np.append( [1], m_vec[0:(2*N)])

    # these will be the coefficients for the x in (1+c_1*x)*(1+c_2*x)*...
    # want coefficients 0, 0, 1, 2, .., 2N-1 - so 2N+1 in total
    coeff_vec = np.append([0], np.arange(0, 2*N))*x + 1

    #this computes the terms involving lambda in a vectorised way
    prod_x_terms_vec = 1/ np.cumprod(coeff_vec)

    #going to create matrix over indices i, j
    delta_mat = np.zeros( (Nplus1, Nplus1) )
    for i in range(Nplus1):
        for j in range(Nplus1):
            # so index is between 0 and 2N, inclusive
            index = i + j
            delta_mat[i,j] = m_vec[index] * prod_x_terms_vec[index]
    return(delta_mat)

def get_partial_products(index, vec):
    return np.prod(vec[:index])

# FIXME: Check index 
def get_index_element(i,j,vec1, vec2):
    index = i+j
    return (vec1[index] * vec2[index])

def det_deltamat_n(x, m_vec, N):
    res = np.linalg.det(deltaNmat_applied(x, m_vec, N))
    # print("res: ",res)
    return res

def get_lambdatilde_p(lambdatilde_1, p, moment_vec, bisect_tol=1e-9):
    lambdatilde_vec = rep(0.0, p)
    lambdatilde_vec[0] = lambdatilde_1
    # print("moment_vec: ",moment_vec)

    if(p>1):
        for i in range(1,p):
            #TODO: uniroot, root
            # print(i) 
            # print("lambdatilde_vec[i-1]: ", lambdatilde_vec[i-1])
            lambdatilde_vec[i] = uniroot(det_deltamat_n, [0, lambdatilde_vec[i-1]], m_vec=moment_vec, N=i+1, tol=bisect_tol)
    lambdatilde_p = lambdatilde_vec[p-1]
    return lambdatilde_p

    
def get_Stilde_polynomial_coefficients(M_p):
    n = M_p.shape[0]
    index = np.arange(1,n+1)
    # TODO: mu_poly_coeff_vec
    mu_poly_coeff_vec = vapply(index, get_ith_coeff_of_Stilde_poly, M_p)

    return mu_poly_coeff_vec

def get_base_vector(n,i):
    base_vec = rep(0, n)
    base_vec[i] = 1
    return base_vec

def get_ith_coeff_of_Stilde_poly(i, mat):
    n = mat.shape[0]
    base_vec = get_base_vector(n,i-1)
    mat[:,n-1] = base_vec
    return (np.linalg.det(mat))

def get_vandermonde(vec):
    '''Generates the van der monde matrix from a vector
    '''
    p = len(vec)
    vdm = np.zeros( (p, p) )
    for i in range(p):
        vdm[i] = vec**i
    return(vdm)

def gen_and_solve_VDM_system(M_p, mu_roots):
    '''Generates the VDM matrix and solves the linear system.
    '''
    b = get_VDM_b_vec(M_p)
    vdm = get_vandermonde(mu_roots)
    # solve the linear system
    pi_vec = np.linalg.solve(vdm, b)
    return(pi_vec)

#simply takes the last column, and removes last element of last column
def get_VDM_b_vec(mat):
    b_vec = mat[:,0]
    # b_vec <- b_vec[-length(b_vec)]
    b_vec = b_vec[:-1]
    return b_vec

#generates the van der monde matrix from a vector
def generate_van_der_monde(vec):
    p = len(vec)
    vdm = np.zeros((p,p))
    for i in range(p):
        vdm[i,:] = (vec**i)
    return vdm

def get_mixed_p_val_vec(quantile_vec, mu_vec, pi_vec, lambdatilde_p):
    p = len(mu_vec)
    alpha = 1/(lambdatilde_p+1e-6)
    beta_vec = mu_vec/alpha
    # pi_vec = np.round(pi_vec, decimals=6)
    # beta_vec = np.round(beta_vec, decimals=6)
    try:
        l = len(quantile_vec)
    except:
        l = 1
    partial_pval_vec = rep(0, l)
    
    for i in range(0,p):
        # TODO: pgamma
        partial_pval_vec = partial_pval_vec + pi_vec[i] * pgamma(quantile_vec, shape=alpha, rate = beta_vec[i])
        
    return partial_pval_vec

def compute_composite_pgamma(index, qval, shape_val, scale_vec):
    return pgamma(qval, shape=shape_val, rate = scale_vec[index])


def get_real_poly_roots(mu_poly_coeff_vec):
    '''Gets real part of complex roots of polynomial with coefficients a,
       where
       a[0] + a[1] * x + ... + a[n-1] * x**(n-1)
       Need to reverse vector to conform with np.roots function
       and then need to reverse again so roos increase in size
    '''
    mu_roots = np.real( np.roots(mu_poly_coeff_vec[::-1]) )
    mu_roots = mu_roots[::-1]
    return(mu_roots)


# lpb4(np.array([1.5, 1.5, 0.5, 0.5]), 10.203)