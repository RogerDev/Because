import numpy as np
# ' Normalizes each column of a matrix to mean zero unit variance.
# ' @param mat Matrix
# ' @return A matrix where each column has mean zero and unit variance.
# ' @examples
# ' normalize(matrix(rnorm(10,2),ncol=2))

# ddof=1 is used to divide by N-1 (due to R code), else by default it is divided by N
# FIXME: is std(x) can be negative??

def func(x):
    if(x.std(ddof=1) > 0):
        return ((x-x.mean())/x.std(ddof=1))
    else:
        return (x-x.mean())


'''
def normalize(mat):
    # if the number of rows is zero
    if(mat.shape[0] == 0):
        mat = mat.T
    mat = np.apply_along_axis(func, 0, mat)
    return mat
'''


def normalize(mat):
    mat = (mat-np.mean(mat, axis=0))/np.std(mat, axis=0)
    return mat
