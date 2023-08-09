import itertools
import random
import pandas as pd
import numpy as np
import scipy.stats as stats
from pprint import pprint
from because.utils import vprint


verbosity = 0
'''
Uniform channel variable class. 
Auto-initializes the distributions for the variable given a base distribution. 
Each parent value(s) has a defined distribution (which may or may not be unique)    
    corresponding to a uniform channel -- each distribution is a permutation of 
    the base distribution. 
Values can be generated from the distributions corresponding to the parent values,
if applicable. 

'''
class UCVar:
    def __init__(self, base_distribution, values=None, parent=None, noise=0):
        self.base_distribution = base_distribution

        if type(values) is list:
            self.values = values
        else:
            self.values = [i for i in range(len(base_distribution))]
        if parent is not None:
            if type(parent) in (UCVar, CategVar) or (type(parent) is list and len(parent) == 1):
                self.distribution = get_dist_dict(parent.values, self.base_distribution, noise)

            elif type(parent) is list and len(parent) > 1:
                self.parent = parent
                # self.parent_size = [parent.size() for parent in self.parent]
                self.parent_vals = [parent.values for parent in self.parent]
                self.distribution = get_dist_dict(self.parent_vals, self.base_distribution, noise)

        else:
            self.distribution = base_distribution

    '''
    Generate a random value from the distribution corresponding to input parent values.
    '''
    def get_value(self, value=None):
        vprint(2, verbosity, f"Value: {value}")
        vprint(2, verbosity, f"Variable distribution: {self.distribution}")
        if value is not None:
            assert type(self.distribution) is dict, "Must have parent variables defined"
            if type(value) is list:
                c = np.random.choice(self.values, p=self.distribution[tuple(value)])
            else:
                c = np.random.choice(self.values, p=self.distribution[value])
        else:
            c = np.random.choice(self.values, p=self.distribution)
        vprint(2, verbosity, f"choice: {c}")
        return c

    def size(self):
        return len(self.base_distribution)


'''
Heuristic variable class for models variables; not well-defined, 
    mainly for testing with UCVar and non-uniform channel distributions. 
Distribution dictionary should be of the form 
    {parent_value_1: [weight_1, weight_2, ..., weight_n], 
    ... , 
    parent_value_m: [weight_1, weight_2, ..., weight_n]}.
'''
class CategVar:
    def __init__(self, distribution, values=None, parent=None):
        self.distribution = distribution
        self.parent = parent
        if values is not None:
            self.values = values
        elif parent is not None:
            self.values = range(len(list(distribution.values())[0]))
        else:
            self.values = range(len(distribution))

    '''
    Generate a random value from the distribution corresponding to input parent values.'''
    def get_value(self, value=None):
        if value is not None:
            assert type(self.distribution) is dict, "Must have parent variables defined"
            if type(value) is list:
                c = np.random.choice(self.values, p=self.distribution[tuple(value)])
            else:
                c = np.random.choice(self.values, p=self.distribution[value])
        else:
            c = np.random.choice(self.values, p=self.distribution)
            vprint(3, verbosity, f"Value choice: {c}")
        return c


'''
Performs the uniform channel test as described in UCM paper, obtaining a p-value for both directions. 
Each parameter A and B should correspond to a list of observations for a single variable. 
Alpha value can be adjusted for confidence, default value is 0.05.
AMap and BMap to remap values to string names; cosmetic and for debugging. 
Returns rho value to determine directionality and confidence, and identifiability boolean. 
'''
def uniform_channel_test(A, B, alpha=0.05, AMap=None, BMap=None, limit=None):
    # if limit is not None and limit > len(A):
    #     idx = random.sample(range(len(A)), 10000)
    #     print(type(A))
    #     A = A[idx]
    #     B = B[idx]

    ab_pval = p_val(A, B, AMap, BMap)
    ba_pval = p_val(A, B, AMap, BMap, reversed=True)
    # ba_pval = p_val(B, A, BMap, AMap)

    vprint(1, verbosity, f"\n\t**P-values** \nForward: {ab_pval} \nBackward:{ba_pval}")

    ''' Alpha may not be needed if we're using a rho value to determine direction; 
        - if both are significant (high magnitude p), then rho goes towards 0. 
        - if neither is signification (low magnitude p), then rho goes towards 0. 
        - Otherwise, one will have a large p value and the other small, so rho is useful. 
    '''

    rho = ab_pval - ba_pval
    vprint(1, verbosity, f"Rho-value: {rho}")

    # Identifiable if tne direction is significant and the other is not
    if (ab_pval > alpha > ba_pval) or (ab_pval < alpha < ba_pval):
        identifiable = True
    else:
        identifiable = False
    vprint(1, verbosity, f"identifiable: {identifiable}")
    return rho, identifiable


'''
Performs the uniform channel test as described in UCM paper, for two variables. 
A and B should each be a list of observations for a particular variable. 
    A[i] and B[i] correspond to the i-th observation of the A and B variables. 
Generates and returns a p-value
'''
def p_val(A, B, AMap=None, BMap=None, reversed = False):

    from scipy.stats.contingency import crosstab
    # if len(A) > 11000:
    #     idx = random.sample(range(len(A)), 10000)
    #     # print(idx)
    #     A = A[idx]
    #     B = B[idx]
    #
    #     # A = A[]:10000
    #     # B = B[:10000]

    if reversed:
        A, B = B, A

    a_size = len(np.unique(A))
    b_size = len(np.unique(B))
    n = len(A)

    # Chunk mostly for seeing conditional prob dist when debugging
    if reversed:
        df = pd.DataFrame({
            'B': A,
            'A': B
        })
        cond_a = (df.groupby('B')['A'].value_counts() / df.groupby('B')['A'].count()).unstack().fillna(0)

        if BMap is not None:
            cond_a.index = [BMap[i] for i in cond_a.index]
        if AMap is not None:
            cond_a.columns = [AMap[i] for i in cond_a.columns]
        vprint(2, verbosity, f"Cond prob Var1|Var2: \n{cond_a}")
    else:
        df = pd.DataFrame({
            'A': A,
            'B': B
        })

        cond_b = (df.groupby('A')['B'].value_counts() / df.groupby('A')['B'].count()).unstack().fillna(0)
        if AMap is not None:
            cond_b.index = [AMap[i] for i in cond_b.index]
        if BMap is not None:
            cond_b.columns = [BMap[i] for i in cond_b.columns]

        vprint(2, verbosity, f"Cond prob Var2|Var1: \n{cond_b}")

    # Obtain the contingency table, sorted rows of observed values
    cont_table = crosstab(A, B)[1]
    cont_table_sort = [sorted(cont_table[i], reverse=True) for i in range(a_size)]
    vprint(4, verbosity, f"Contingency table: \n{cont_table_sort}")

    # Calculate MLE from columns of the sorted contingency table
    cat_counts = [[cont_table_sort[i][j] for i in range(a_size)] for j in range(b_size)]  # Group by size category
    mle = [sum(i) / n for i in cat_counts]

    # Obtain the expected values (sorted table)
    N_a = [sum(cont_table[i]) for i in range(a_size)]
    exp_cont_table = [[N_a[i] * mle[j] for j in range(b_size)] for i in range(a_size)]
    vprint(4, verbosity, f"Expected table: \n{exp_cont_table}")

    # Calculate a G-score
    summand = [cont_table_sort[x][y] * np.log((cont_table_sort[x][y] / (exp_cont_table[x][y] + 0.000000001)) + 0.000000001)
               for x in range(a_size) for y
               in range(b_size)]
    vprint(5, verbosity, '\tSummand: {summand}')
    g_squared = 2 * sum(summand)
    vprint(5, verbosity, "\tg-squared: {g_squared}")

    # Calculate the corresponding p-value from g-squared statistic
    dof = (a_size - 1) * (b_size - 1)
    p_value = 1 - stats.chi2.cdf(g_squared, df=dof)

    return p_value


# For initializing noisy UC probability distributions within models; Not really needed anymore
def get_noisy_perm(base_dist, noise):
    # print(base_dist)
    # print(sum(base_dist))
    # assert sum(base_dist) == 1, "Distribution must sum to 1"

    from random import uniform, seed
    from numpy.random import choice
    from itertools import permutations
    # seed(1)

    # Generate noise
    noise = [uniform(0, noise) for i in range(len(base_dist))]

    # Add noise to a permutation
    perms = list(permutations(base_dist))
    unnorm_dist = np.array(perms[choice(range(len(perms)))]) + noise

    # Normalize distribution
    norm_dist = [float(i) / sum(unnorm_dist) for i in unnorm_dist]

    # print(norm_dist)

    return norm_dist

# Auto-generates a (UNIFORM CHANNEL) distribution dictionary dependent on possible parent values and base distribution
# For generating models variable values
def get_dist_dict(parent_vals, base_dist, noise=0.0):
    dist_dict = dict()
    # multiple parents
    if type(parent_vals[0]) is list:
        prod = itertools.product(*parent_vals)
        for elem in prod:
            dist_dict[elem] = get_noisy_perm(base_dist, noise)
    else:  # single parent
        for val in parent_vals:
            dist_dict[val] = get_noisy_perm(base_dist, noise)
    # if verbosity >= 3:
    #     pprint(dist_dict)
    return dist_dict



if __name__ == "__main__":

    # for i in range(1):
    #     print(f"i =  {i}")
    #     # d = get_noisy_perm([0.1, 0.1, 0.8])
    #     get_dist_dict(2, [0.1, 0.1, 0.8])
    #     # print(np.random.choice(range(3), p=d))

    # CatA = UCVar([0.1, 0.1, 0.8])
    # CatC = UCVar([0.4, 0.6])
    # CatB = UCVar([0.2, 0.8], parent=[CatA, CatC])


    # for i in range(10):
    #     A = CatA.get_value()
    #     C = CatC.get_value()
    #     # print(A)
    #     B = CatB.get_value([A, C])

    CatB = CategVar([0.4, 0.5, 0.1])

    CatA = UCVar([0.25, 0.25, 0.5], parent=CatB)
    CatC = UCVar([0.2, 0.8], parent=[CatA, CatB])
    for i in range(10):
        B = CatB.get_value()
        A = CatA.get_value(B)
        # print(A)
        C = CatC.get_value([A, B])
        print("VALUES: ", A, B, C)


