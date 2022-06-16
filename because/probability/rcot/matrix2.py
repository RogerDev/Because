#' Converts a vector to a matrix, or does nothing if the input is a matrix.
#' @param mat Vector or matrix.
#' @return Matrix.
#' @examples
#' matrix2(rnorm(10));
#' matrix2(matrix(rnorm(10,2),ncol=2));
 # matrix() in R changes the row vector to column vector and does nothing if it is already a column vector

def matrix2(mat):
    if(mat.shape[0] == 1):
        return mat.T
    return mat
        