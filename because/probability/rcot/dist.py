import numpy as np
# this gives a matrix of the distances between any two rows


def dist(mat):
    # return c(t(dist(x[1:r1,])))
    ans = []
    for i in range(mat.shape[0]):
        for j in range(i+1, mat.shape[0]):
            ans.append(np.linalg.norm(mat[i, :] - mat[j, :]))
    return np.array(ans)