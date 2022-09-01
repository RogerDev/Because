from because.probability import ProbSpace

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
        # Now generate first level tests
        if minLevel == 0:
            for pair in pairs:
                yield (pair[0], pair[1], [])                
        if maxLevel >= 1 and minLevel <= 1:
            for pair in pairs:
                # Generate tests with one conditional
                for i in range(len(targets)):
                    interm = targets[i]
                    if interm == pair[0] or interm == pair[1]:
                        continue
                    yield (pair[0], pair[1], [interm])
        if maxLevel >= 2 and minLevel <= 2:
            # And do the same for any double conditionals
            for pair in pairs:
                for i in range(len(targets)):
                    interm = targets[i]
                    if interm == pair[0] or interm == pair[1]:
                        continue
                    for j in range(i+1, len(targets)):
                        interm2 = targets[j]
                        if interm2 == pair[0] or interm2 == pair[1]:
                            continue
                        yield (pair[0], pair[1], [interm, interm2])
        if maxLevel >= 3 and minLevel <= 3:
            # Now do 3 level conditionals
            for pair in pairs:
                for i in range(len(targets)):
                    interm = targets[i]
                    if interm == pair[0] or interm == pair[1]:
                        continue
                    for j in range(i+1, len(targets)):
                        interm2 = targets[j]
                        if interm2 == pair[0] or interm2 == pair[1]:
                            continue
                        for k in range(j+1, len(targets)):
                            interm3 = targets[k]
                            if interm3 == pair[0] or interm3 == pair[1]:
                                continue
                            yield (pair[0], pair[1], [interm, interm2, interm3])

        if maxLevel >= 4 and minLevel <= 4:
            # Now do 4 level conditionals
            for pair in pairs:
                for i in range(len(targets)):
                    interm = targets[i]
                    if interm == pair[0] or interm == pair[1]:
                        continue
                    for j in range(i+1, len(targets)):
                        interm2 = targets[j]
                        if interm2 == pair[0] or interm2 == pair[1]:
                            continue
                        for k in range(j+1, len(targets)):
                            interm3 = targets[k]
                            if interm3 == pair[0] or interm3 == pair[1]:
                                continue
                            for l in range(k+1, len(targets)):
                                interm4 = targets[l]
                                if interm4 == pair[0] or interm4 == pair[1]:
                                    continue
                                yield (pair[0], pair[1], [interm, interm2, interm3, interm4])
                                
def calculateAll(ps, testList=None, varNames=None, minLevel=0, maxLevel=3, sensitivity=5, power=5):
    if testList is None:
        assert varNames is not None, 'calc_indeps.calculate: must provide testList or varNames.'
        testList = getTestItems(varNames, minLevel=minLevel, maxLevel=maxLevel)
    for item in testList:
        v1 = item[0]
        v2 = item[1]
        conds = item[2]
        if ps.isIndependent(v1, v2, conds, power=power, sensitivity=sensitivity):
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
        if ps.isIndependent(v1, v2, conds, power=power, sensitivity=sensitivity):
            yield(item, True)
        else:
            yield(item, False)
    
if __name__ == '__main__':
    print('testItems  = ')
    testItems = getTestItems(['X1', 'X2', 'X3', 'X4', 'X5', 'x6'], maxLevel=0)
    for testItem in testItems:
        print('   ', testItem)