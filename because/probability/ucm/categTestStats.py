import pandas as pd
import numpy as np
import math
import scipy.stats as stats
'''
Testing file for developing UCM implementation. 
'''

# df = pd.read_csv("models/trinaryCateg.csv")
# df = pd.read_csv("models/categTest1.csv")
df = pd.read_csv("models/M1C.csv")

n = len(df)
print(f"\n Number of entries: {len(df)}")

x_size = 2
y_size = 2

# ---- Joint Probability
print("\n\tJoint Probability Table")
joint_prob = df.value_counts(["A", "B"]).unstack()
joint_rows = [joint_prob.iloc[i].to_list() for i in range(x_size)]
print(joint_prob)

# ---- Y|X Conditional Probability (Causal Direction)
cond_y = (df.groupby('A')['B'].value_counts() / df.groupby('A')['B'].count()).unstack().T
cond_y_trunc = cond_y.round(3)  # truncated probabilities
y_rows = [cond_y_trunc.iloc[i].to_list() for i in range(x_size)]  # each row is prob y given X = i
y_sort = [sorted(y_rows[i], reverse=True) for i in range(x_size)]
#
print(f"\n\tProbability of Y given X: \n {cond_y}")
print(f"\n\t Rows of Y|X cond prob: \n")
print(*y_rows, sep='\n')
print(f"\n\t Sorted rows: \n")
print(*y_sort, sep='\n')

# ---- X|Y Conditional Probability (NonCausal Direction)
cond_x = (df.groupby('B')['A'].value_counts() / df.groupby('B')['A'].count()).unstack()
cond_x_trunc = cond_x.round(3)  # truncated probabilities
x_rows = [cond_x_trunc.iloc[i].to_list() for i in range(y_size)]
x_sort = [sorted(x_rows[i], reverse=True) for i in range(y_size)]

print(f"\n\tProbability of X given Y: \n{cond_x}")
print(f"\n\t Rows of X|Y cond prob: \n")
print(*x_rows, sep='\n')
print(f"\n\t Sorted rows: \n")
print(*x_sort, sep='\n')

# ---- Calculate MLE for the rows
# Test one: calculate MLE for y given x=1.
# Entry 1: N(Y=1|X=1) / N(X=1)
print("\n\tContingency table")
cont_table = df.value_counts(['B', 'A']).unstack()
print(cont_table)
cond_y = (df.groupby('A')['B'].value_counts() / df.groupby('A')['B'].count()).unstack()
x_row_count = df['A'].value_counts()  # this returns the sum of each X row
y_col_count = df['B'].value_counts()  # this returns the sum of each Y column

row_count = [cont_table.iloc[i].to_list() for i in range(x_size)]
row_count_sort = [sorted(row_count[i], reverse=True) for i in range(x_size)]
print(f"\n\tSorted rows:")
print(*row_count_sort, sep="\n")

# --- Calculating the MLE for the distribution of Y given X.
# --- Use the MLE to determine how different each row is from the predicted distribution (for a uniform channel)

# Sort row counts by largest category in each row to smallest.
t = [[row_count_sort[i][j] for i in range(x_size)] for j in range(y_size)]
N_x = [sum(row) for row in row_count]
print(t)
print(N_x)

expected_xy = [[(t[y][x], N_x[x]) for x in range(x_size)] for y in range(y_size)]
print(f"Expected xy: {expected_xy}")
# p = [(row_count_sort[x][y]*N_x[x]/n) for y in range(y_size) for x in range(x_size)]
# print(p)
# # mle = sum(t)/len(df)
mle = [sum(i) / n for i in t]
# # mle = expected_xy/n
print(f"\n\tMLE: {mle}")


# New test: try to do the entropy calculation for each row. Compare to the entropy of MLE.
def calc_entropy(row):
    entropy = -1 * sum(elem * np.log(elem+0.000001) for elem in row)
    return round(entropy,4)


mle_entropy = calc_entropy(mle)
observed_entropy = [calc_entropy(row) for row in y_rows]
print(f"MLE: {mle} \n\t Entropy: {mle_entropy}")
print(f"Observed: {y_rows} \n\t Entropy: {observed_entropy}")

print(calc_entropy([1 / 3, 1 / 3, 1 / 3]))
print(calc_entropy([0, 0, 1]))

# # Calculate expected value for each row: using the MLE just distribute each row
# exp_xy_sorted = [[N_x[i] * mle[j] for j in range(y_size)] for i in range(x_size)]
# print("\n\tExpected sorted rows:")
# print(*exp_xy_sorted, sep="\n")
#
# # Compare sorted rows to sorted expected rows
# # p = [(exp_xy_sorted[x][y], row_count_sort[x][y]) for x in range(x_size) for y in range(y_size)]
# pre_g = [row_count_sort[x][y]*np.log(row_count_sort[x][y]/exp_xy_sorted[x][y]) for x in range(x_size) for y in range(y_size)]
# g_squared = 2*sum(pre_g)
# print(f"\n\tG-squared value: {g_squared}")
#
# # Convert to a real p value
# dof = (x_size-1) * (y_size-1)
# print(f"p-value: {round(1-stats.chi2.cdf(g_squared, df=dof), 4)}")
#

# # Repeat for non-causal direction
# print("\n\tContingency table (Y->X)")
#
# cont_table = df.value_counts(['Y', 'X']).unstack()
# print(cont_table)
# row_count = [cont_table.iloc[i].to_list() for i in range(y_size)]
# row_count_sort = [sorted(row_count[i], reverse=True) for i in range(y_size)]
#
# t = [[row_count_sort[i][j] for i in range(y_size)] for j in range(x_size)]
# N_y = [sum(row) for row in row_count]
# expected_yx = [[(t[x][y], N_y[y]) for y in range(y_size)] for x in range(x_size)]
# mle = [sum(i)/n for i in t]
# exp_yx_sorted = [[N_y[i] * mle[j] for j in range(x_size)] for i in range(y_size)]
# print(*exp_yx_sorted, sep="\n")
# pre_g = [row_count_sort[y][x]*np.log(row_count_sort[y][x]/exp_yx_sorted[y][x]) for y in range(y_size) for x in range(x_size)]
# print("COMPARE:", [(row_count_sort[y][x], exp_yx_sorted[y][x]) for y in range(y_size) for x in range(x_size)])
# g_squared = 2*sum(pre_g)
# print(f"g-squared: {g_squared}")
# dof = (x_size-1) * (y_size-1)
# print(f"p-value: {1-stats.chi2.cdf(g_squared, df=dof)}")


# # Obtain expected value and observed value as pairs
# oe_pairs = [(t[i][j], expected_xy[i]) for i in range(y_size) for j in range(x_size)]
# print(f"Observed/Expected pairs: {oe_pairs}")
# print("1", [(t[i][j]/expected_xy[i]) for i in range(y_size) for j in range(x_size)])
# print("2", [np.log(t[i][j]/expected_xy[i]) for i in range(y_size) for j in range(x_size)])
# print("3", [t[i][j] * np.log(t[i][j]/expected_xy[i]) for i in range(y_size) for j in range(x_size)])

# Do G-test to obtain p value
# w = [t[i][j] * np.log(t[i][j]/expected_xy[i]) for i in range(y_size) for j in range(x_size)]
# print(w)
# g_squared = 2*sum(w)
# print(g_squared)
# dof = (x_size-1) * (y_size-1)

# print(f"p value: {1-stats.chi2.cdf(g_squared, df=dof)}")


# --- Compare MLE to row computations
print("\n\tTotal difference from sorted rows to MLE")
for i in range(x_size):
    # Get % difference from MLE
    print(f"\tRow {i}:")
    s = [round(mle[j] - y_sort[i][j], 4) for j in range(y_size)]
    print(s)

print("\n\tPercent difference from sorted rows to MLE")
for i in range(x_size):
    # Get % difference from MLE
    print(f"\tRow {i}:")
    s = [round((mle[j] - y_sort[i][j]) / mle[j], 4) for j in range(y_size)]
    print(s)

# --- Repeat for Y->X direction
print("\n----- Y -> X -----")
print("\n\tContingency table")
cont_table = df.value_counts(['B', 'A']).unstack()
print(cont_table)
row_count = [cont_table.iloc[i].to_list() for i in range(y_size)]
row_count_sort = [sorted(row_count[i], reverse=True) for i in range(y_size)]
print(row_count_sort)

t = [[row_count_sort[i][j] for i in range(y_size)] for j in range(x_size)]
print(t)
mle = [sum(i) / n for i in t]
print(mle)

mle_entropy = calc_entropy(mle)
observed_entropy = [calc_entropy(row) for row in x_rows]
print(f"MLE: {mle} \n\t Entropy: {mle_entropy}")
print(f"Observed: {x_rows} \n\t Entropy: {observed_entropy}")

expected_yx = [sum(i) / y_size for i in t]
print(f"Expected yx vals: {expected_yx}")

mle = [sum(i) / n for i in t]
print(f"\n\tMLE: {mle}")
#
# # Obtain expected value and observed value as pairs
# oe_pairs = [(t[i][j], expected_yx[i]) for i in range(x_size) for j in range(y_size)]
# print(f"Observed/Expected pairs: {oe_pairs}")
#
# # Do G-test to obtain p value
# w = [t[i][j] * np.log(t[i][j]/expected_yx[i]) for i in range(x_size) for j in range(y_size)]
# g_squared = 2*sum(w)
# print(g_squared)
# dof = (x_size-1) * (y_size-1)
#
# print(f"p value: {1-stats.chi2.cdf(g_squared, df=dof)}")

# --- Compare MLE to row computations
print("\n\tTotal difference from sorted rows to MLE")
for i in range(y_size):
    # Get % difference from MLE
    print(f"\tRow {i}:")
    s = [round(mle[j] - x_sort[i][j], 4) for j in range(x_size)]
    print(s)

print("\n\tPercent difference from sorted rows to MLE")
for i in range(y_size):
    # Get % difference from MLE
    print(f"\tRow {i}:")
    s = [round((mle[j] - x_sort[i][j]) / mle[j], 4) for j in range(x_size)]
    print(s)

