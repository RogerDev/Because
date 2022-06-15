from .matrix2 import matrix2
import numpy as np
# ' Repeat rows or columns of a vector or matrix
# ' @param X Vector or matrix
# ' @param m number of row copies
# ' @param n number of column copies
# ' @return Matrix.
# ' @examples
# ' repmat(matrix(rnorm(10*2),ncol=2),2,2)


def repmat(X, m, n):
    X = matrix2(X)
    mx = X.shape[0]
    try:
        nx = X.shape[1]    # D => dimension of variable
    except:
        nx = 1
    # matrix(t(matrix(X,mx,nx*n)),mx*m,nx*n,byrow=T)
    X = X.flatten(order='F')
    X1 = np.array([X[i % len(X)] for i in range(mx*nx*n)])
    X1 = X1.reshape(nx*n, mx, order='F')
    X1 = X1.flatten(order='F')
    X2 = np.array([X1[i % len(X1)] for i in range(mx*m*nx*n)])
    X2 = X2.reshape(mx*m, nx*n, order='C')
    return X2
