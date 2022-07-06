import numpy as np
import pandas as pd
from .matrix2 import matrix2
from .normalize import normalize
from .random_fourier import random_fourier_features
from .dist import dist
from .expandgrid import expandgrid
from .lpb4 import lpb4
from .hbe import hbe

#' Tests whether x and y are unconditionally independent
#' @param x Random variable x.
#' @param y Random variable y.
#' @param approx Method for approximating the null distribution. Options include:
#' "lpd4," the Lindsay-Pilla-Basak method (default),
#' "gamma" for the Satterthwaite-Welch method,
#' "hbe" for the Hall-Buckley-Eagleson method,
#' "chi2" for a normalized chi-squared statistic,
#' "perm" for permutation testing (warning: this one is slow but recommended for small samples generally <500 )
#' @param seed The seed for controlling random number generation. Use if you want to replicate results exactly. Default is NULL.
#' @return A list containing the p-value \code{p} and statistic \code{Sta}
#' @examples
#' RIT(rnorm(1000),rnorm(1000));
#'
#' x=rnorm(1000);
#' y=(x+rnorm(1000))^2;
#' RIT(x,y);

def colMeans(vec):
    vec = np.array(vec)
    return np.mean(vec, axis=0)

def RIT(x,y,num_f2=5,approx="lpd4", seed=None):

    x = np.matrix(x).T    
    y = np.matrix(y).T

    if(np.std(x) == 0 or np.std(y) == 0):
        return ([1.0], 0)   # this is P value
    
    x = matrix2(x)
    y = matrix2(y)

    #r1 = x.shape[0]
    #if (r1 > r):
    #    r1 = r
    r = x.shape[0]

    x = normalize(x)
    y = normalize(y)

    #(four_x, w, b) = random_fourier_features(x,num_f=num_f2,sigma=np.median(dist(x[:r1, ])), seed = seed )
    #(four_y, w, b) = random_fourier_features(y,num_f=num_f2,sigma=np.median(dist(y[:r1, ])), seed = seed )
    (four_x, w, b) = random_fourier_features(x, num_f=num_f2, sigma=1, seed=seed)
    (four_y, w, b) = random_fourier_features(y, num_f=num_f2, sigma=1, seed=seed)

    f_x = normalize(four_x)
    f_y = normalize(four_y)

    Cxy = np.cov(f_x, f_y, rowvar=False)
    Cxy = Cxy[:num_f2, num_f2:]  # num_f2,num_f2

    Sta = r*np.sum(Cxy**2)

    # res_x = f_x-repmat(t(matrix(colMeans(f_x))),r,1);
    res_x = f_x - np.repeat(np.matrix(colMeans(f_x))[:,np.newaxis],r,axis=1)
    # res_y = f_y-repmat(t(matrix(colMeans(f_y))),r,1);
    res_y = f_y - np.repeat(np.matrix(colMeans(f_x))[:,np.newaxis],r,axis=1)

    # d =expand.grid(1:ncol(f_x),1:ncol(f_y));
    d = expandgrid(np.arange(0,f_x.shape[1]), np.arange(0,f_y.shape[1]))
    # res = res_x[,d[,1]]*res_y[,d[,2]];
    # print(d)
    # print("type x: ",type(res_x))
    # print("type y: ",type(res_y))
    res = np.array(res_x[:,np.array(d['Var1'])]) * np.array(res_y[:,np.array(d['Var2'])])
    res = np.matrix(res)
    #print("res: ",res)
    #print(res.shape)
    # Cov = 1/r * (t(res)%*%res);
    Cov = 1/r * ((res.T) @ res)
    #print("Cov: ",Cov)
    #print(Cov.shape)
    # eig_d = eigen(Cov);
    w,v = np.linalg.eig(Cov)
    w = w.real
    # eig_d$values=eig_d$values[eig_d$values>0];
    w = [i for i in w if i>0]
    #print("eig_d$values: ",w)
    # print(w.shape)

    # if (approx == "lpd4"){
    #   eig_d_values=eig_d$values;
    #   p=try(1-lpb4(eig_d_values,Sta), silent=TRUE);
    #   if (!is.numeric(p)  | is.nan(p)){
    #     p=1-hbe(eig_d$values,Sta);
    #   }
    # }
    if(approx == "lpd4"):
        w1 = w
        try:
            p = 1 - lpb4(np.array(w1), Sta)
        except ValueError:
            p = [1 - hbe(np.array(w1), Sta)]
        if(p==None or np.isnan(p)):
            p = [1 - hbe(np.array(w1), Sta)]

    #print("Sta: ",Sta)
    #print("p: ",p)
    return (p, Sta)

if __name__ == '__main__':
    data = pd.read_csv("M2.csv")
    # print(data)
    x = data['X']
    y = data['Y']
    # print(x)
    # print(y)
    # print(z)
    (p, Sta) = RIT(x, y)

