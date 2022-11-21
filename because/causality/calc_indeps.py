from because.probability import ProbSpace
from because.probability.utils import getCombos

def getTestItems(targets, testPair=None, minLevel=0, maxLevel=3):
        assert minLevel <= maxLevel, "calc_indeps.getTestItems: minLevel must be less than or equal to maxLevel (min, max) = " + str(minLevel) + ', ' + str(maxLevel)
        pairs = []
        if testPair is None:
            # We're testing all pairs of variables
            for i in range(len(targets)):
                for j in range(i+1, len(targets)):
                    pair = (targets[i], targets[j])
                    pairs.append(pair)
        else:
            # Only testing one pair
            pairs.append(testPair)
        # At this point we have all unordered pairs of variables
        # Now extend that with all conditional at each rank.
        for rank in range(minLevel, maxLevel + 1):
            for pair in pairs:
                v1, v2 = pair
                conds = []
                for v in targets:
                    if v == v1 or v == v2:
                        continue
                    conds.append(v)
                combos = getCombos(conds, minRank=minLevel, maxRank = maxLevel)
                for combo in combos:
                    yield((v1, v2, combo))

                                
def calculateAll(ps, testList=None, varNames=None, minLevel=0, maxLevel=3, sensitivity=5, power=5):
    if testList is None:
        assert varNames is not None, 'calc_indeps.calculate: must provide testList or varNames.'
        testList = getTestItems(varNames, minLevel=minLevel, maxLevel=maxLevel)
    for item in testList:
        v1 = item[0]
        v2 = item[1]
        conds = item[2]
        num_f = 100 if len(conds) == 0 else 100 * len(conds)
        if ps.isIndependent(v1, v2, conds, power=power, sensitivity=sensitivity, num_f=num_f, seed=1):
            yield(item, True)
        else:
            yield(item, False)

def calculateOne(ps, v1, v2, varNames, minLevel=0, maxLevel=3, sensitivity=5, power=5):
    pair = (v1, v2)
    testList = getTestItems(varNames, testPair = pair, minLevel=minLevel, maxLevel=maxLevel)
    for item in testList:
        v1 = item[0]
        v2 = item[1]
        conds = item[2]
        #print('calculateOne: v1, v2, conds = ', v1, v2, conds)
        num_f = 100 if len(conds) == 0 else 100 * len(conds)
        if ps.isIndependent(v1, v2, conds, power=power, sensitivity=sensitivity, num_f=num_f, seed=1):
            yield(item, True)
        else:
            yield(item, False)
    
if __name__ == '__main__':
    print('testItems  = ')
    testItems = getTestItems(['X1', 'X2', 'X3', 'X4', 'X5', 'x6'], maxLevel=0)
    for testItem in testItems:
        print('   ', testItem)