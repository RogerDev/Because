#TODO: check all()
def alls(vec):
    return all(x>0 for x in vec)

def checkCoeffsArePositiveError(coeff):
    if(len(coeff) == 0):
        return True
    
    if(not alls(coeff)):
        return True
    return False

def getCoeffError(coeff):
    if(len(coeff) == 0):
        return "empty coefficient vector."

    if(not alls(coeff)):
        return "not all coefficients > 0."
    
    return "unknown error."

def checkXvaluesArePositiveError(x):
    if(len(x) == 0):
        return True
    
    if(not alls(x)):
        return True
    
    return False

def getXvaluesError(x):
    if(len(x) == 0):
        return "empty x vector."

    if(not alls(x)):
        return "not all x-values > 0."

    return "unknown error."