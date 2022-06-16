def pgamma(q,shape,rate):
    """
    Calculates the cumulative of the Gamma-distribution
    """
    from scipy.stats import gamma    
    result=gamma.cdf(x=q,a=shape,loc=0,scale=rate)
    return result