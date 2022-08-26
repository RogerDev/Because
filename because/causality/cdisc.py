from ssl import CHANNEL_BINDING_TYPES
from because.causality import calc_indeps
from because.causality import rv

def discover(ps, varNames=None, maxLevel=2, power=5):
    if varNames is None:
        varNames = ps.getVarNames()
    links = []
    # First identify causal links
    for i in range(len(varNames)):
        v1 = varNames[i]
        for j in range(i+1, len(varNames)):
            valid = True
            v2 = varNames[j]
            indeps = calc_indeps.calculateOne(ps, v1, v2, varNames=varNames, maxLevel=maxLevel, power=power)
            for indep in indeps:
                # If we find any independencies, direct or conditional, mark it as not causal.
                spec, isInd = indep
                if isInd:
                    valid = False
                    break
            if valid:
                # We have a causal link. Check the direction and add it.
                dir = ps.testDirection(v1, v2, power=power, N_train=10000)
                if dir > 0:
                    links.append((v1, v2))
                else:
                    links.append((v2, v1))
                print('found causal link:', links[-1])
    # Now turn it into an RV model.
    varDict = {}
    for var in varNames:
        varDict[var] = []
    for link in links:
        child = link[1]
        parent = link[0]
        varDict[child].append(parent)
    gnodes = []
    for var in varNames:
        parents = varDict[var]
        if ps.isCategorical(var):
            dType = rv.RVType.CATEGORICAL
        else:
            dType = rv.RVType.NUMERIC
        gnode = rv.RV(var, parents, True, dType, None, None)
        gnodes.append(gnode)
    return gnodes



    