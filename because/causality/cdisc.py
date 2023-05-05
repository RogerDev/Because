
import time
from math import log

from because.causality import calc_indeps
from because.causality import rv
from because.causality import cgraph
import networkx as nx
from because.probability.utils import getCombos
from because.utils import vprint

def resolveLoops(loops, links, verbosity):
    resolutions = []
    resDict = {}
    for loop in loops:
        for i in range(0, len(loop)):
            if i == 0:
                # The pair is the wrap (i.e (last,first))
                var1 = loop[-1]
            else:
                var1 = loop[i-1]
            var2 = loop[i]
            for link in links:
                v1, v2, dir = link
                if var1 == v1 and var2 == v2:
                    if (var1, var2) in resDict:
                        dir, link = resDict[(var1, var2)]
                        cnt += 1
                    else:
                        cnt = 1
                    resDict[(var1, var2)] = (dir, link)
    resolutions = list(resDict.values())
    resolutions.sort()
    # Reverse the link with the most loops and smallest direction
    dir, rlink = resolutions[0]
    outlinks = []
    for link in links:
        if link == rlink:
            v1, v2, dir = link
            # Give it a high direction so it won't get reversed again.
            if verbosity >= 3:
                print('reversing link = ', link)
            outlinks.append((v2, v1, dir))
        else:
            outlinks.append(link)
    return outlinks

def MVACSet(ps, v1, v2, adjSet, verbosity=0):
    """
    Calculate the Minimum Variance Adjacency Conditional (MVAC) Set, given a 
    pair of variables, v1 and v2, and a set of adjacencies (adjSet).
    ps is a ProbSpace instance.
    Returns the subset of adjSet that, when conditioned on, yields the lowest
    combined variance for v1 and v2 in the form of a conditional specification:
    [(varName, valueTotest), ...]
    """
    minVar = 0.0
    mvacSet = []
    avar = ps.distr(v1).stDev()**2
    bvar = ps.distr(v2).stDev()**2
    totvar = avar + bvar
    minVar = totvar
    if verbosity >= 3:
        print('MVACSet: v1, v2, totvar = ', v1, v2, totvar)

    varModes = {}
    for v in adjSet:
        varModes[v] = ps.distr(v).mode()
    combos = getCombos(adjSet, len(adjSet))

    for c in combos:
        cSpec = []
        # Form a hierarchical query with maximum likeihood.  For example:
        # [A, B, C] => [(A, mode(A)), (B, mode(B | A=mode(A))), 
        #              (C, mode(C) | A=mode(A), B=mode(B | A=mode(A)))]
        for i in range(len(c)):
            v = c[i]
            # if i == 0:
            #     cSpec.append((v, varModes[v]))
            # else:
            #     key = tuple(c[:i+1])
            #     if key not in varModes:
            #         mode = ps.distr(v, cSpec).mode()
            #         varModes[key] = mode
            #     else:
            #         mode = varModes[key]
            #     cSpec.append((v, mode))
            cSpec.append((v, varModes[v]))
        adist = ps.distr(v1, cSpec)
        if adist is None or adist.N < 10:
            # Not enough points for query.  Skip this one.
            if verbosity >= 1:
                print('MVACSet: Too few points to condition on. Skipping:', v1, cSpec)
            continue
        bdist = ps.distr(v2, cSpec)
        if bdist is None or bdist.N < 10:
            # Not enough points for query.  Skip this one.
            if verbosity >= 1:
                print('MVACSet: Too few points to condition on. Skipping:', v2, cSpec)
            continue
        avar = adist.stDev()**2
        bvar = bdist.stDev()**2
        totvar = avar + bvar
        if verbosity >= 3:
            print('MVACSet: v1, v2, combo, cspec, totvar = ', v1, v2, c, cSpec, totvar)
        if totvar < minVar:
            minVar = totvar
            mvacSet = cSpec
    return mvacSet


def testDirection_old(ps, v1, v2, adjSet, power=5, verbosity=0):
    mvacs = MVACSet(ps, v1, v2, adjSet, verbosity=verbosity)
    print('mvacs = ', mvacs)
    ss = ps.SubSpace(mvacs, minPoints=500, maxPoints=1000)
    rho = ss.testDirection(v1, v2, power=power)
    print('ss.N, rho = ', ss.N, rho)
    return rho

def h0(p):
    if p > 0:
        return p * log(1/p)
    else:
        return 0
    
def h(hist):
    cum = 0
    for p in hist:
        cum += h0(p)
    return cum
def dirDE(ps, rv1, rv2, verbosity=0):
    d1 = ps.distr(rv1)
    d2 = ps.distr(rv2)
    hist1 = d1.ToHistogram()
    hist2 = d2.ToHistogram()
    h1 = h(hist1)
    h2 = h(hist2)
    v1 = ps.getValues(rv1)
    v2 = ps.getValues(rv2)
    maxEnt1 = h([1/len(v1)] * len(v1))
    maxEnt2 = h([1/len(v2)] * len(v2))
    h1_s = h1 / maxEnt1
    h2_s = h2 / maxEnt2
    de = (h1_s - h2_s) / (h2_s + h1_s)
    vprint(1, verbosity, 'cdisc.dirDE: entropies', rv1, rv2, '=', h1,h2, h1_s, h2_s, de)
    return de

def testDirection(ps, v1, v2, adjSet, power=5, maxLevel=None, verbosity=0):
    """
    Test the directionality with all combinations of conditionalities with
    adjacent variables.  Choose the rho with the highest absolute value.
    """
    if maxLevel is None:
        maxLevel = len(adjSet)
    varModes = {}
    for v in adjSet:
        varModes[v] = ps.distr(v).mode()
    maxAbsRho = 0
    maxRho = 0
    bestCombo = None
    bestN = 0
    combos = getCombos(adjSet, maxLevel, 0) # Include null combo
    cum = 0.0
    for c in combos:
        cSpec = []
        for i in range(len(c)):
            v = c[i]
            if i == 0:
                 cSpec.append((v, varModes[v]))
            else:
                 key = tuple(c[:i+1])
                 if key not in varModes:
                     mode = ps.distr(v, cSpec).mode()
                     varModes[key] = mode
                 else:
                     mode = varModes[key]
                 cSpec.append((v, mode))
        rho = ps.testDirection(v1, v2, givenSpecs=cSpec, power=power)
        cum += rho
        if abs(rho) > maxAbsRho:
            maxAbsRho = abs(rho)
            maxRho = rho
            bestCombo = c
        if verbosity >= 4:
            print('        cdisc.testDirection: v1, v2, combo, rho, N = ', v1, v2, c, rho)
    if verbosity >= 3:
        print('      cdisc.testDirection: best combo for(', v1, ',', v2 , ') = ', bestCombo, ', rho = ', maxRho)
    outRho = cum / len(combos)
    if verbosity >= 3:
        print('       cdisc.testDirection: avgRho = ', outRho)
    return outRho

def diffEntropy(ps, a, b):
    d1 = ps.distr(a)
    d2 = ps.distr(b)
    hist1 = d1.ToHistogram()
    hist2 = d2.ToHistogram()
    h1 = 0
    h2 = 0
    for i in range(len(hist1)):
        p = hist1[i]
        h1 += p * log(1/p) / len(hist1)
    for i in range(len(hist2)):
        p = hist2[i]
        h2 += p * log(1/p) / len(hist2)

    print('entropy for', a, b, 'is', h1, h2)    
    de = (h1 - h2) / (h1 + h2)
    return de

def discover(ps, varNames=None, maxLevel=2, power=5, sensitivity=5, verbosity=2):
    """
    Explore the given dataset (ProbSpace instance) and develop a Causal Model, given
    the hyper-parameters power and sensitivity.
    Parameters:
        power determines how much of the dataset is used for each test.  It allows a tradeoff
            between runtime and accuracy. Range [0,100].  Recomended values [3,10].  Default is 5.
        sensitivity determines the threshold for independence testing, allowing more or less sensitivity
            to dependence. Range [0,10].  Default 5.  In general, higher sensitivity will result in more
            links, and lower sensitivity in fewer links in the causal graph.
    """
    start = time.time()
    if varNames is None:
        varNames = ps.getVarNames()
    
    blocked = {} # Ditionary of variable pair to blocking list, for any links blocked by an intermediary
    nonmed = {} # Dictionary of variable pair to invalid mediaries (i.e. intermediaries).
    # Process one level (i.e. number of dependencies) at a time.
    for level in range(maxLevel + 1):
        independencies = []
        if verbosity >= 2:
            print('Pass', level+1, 'of', maxLevel+1)
        for i in range(len(varNames)):
            v1 = varNames[i]
            for j in range(i+1, len(varNames)):
                v2 = varNames[j]
                if (v1, v2) in blocked:
                    # This link is already blocked
                    continue
                mediaries = [] # Possible mediaries
                # Remove any invalid mediaries from consideration
                nm1 = nonmed.get((v1, v2), [])
                nm2 = nonmed.get((v2, v1), [])
                nm = nm1 + nm2
                for varName in varNames:
                    if varName not in nm and varName != v1 and varName != v2:
                        mediaries.append(varName)
                indeps = calc_indeps.calculateOne(ps, v1, v2, varNames=mediaries,
                            minLevel=level, maxLevel=level, power=power, sensitivity=sensitivity)
                for indep in indeps:
                    # If we find any independencies, direct or conditional, mark it as not causal.
                    spec, isInd = indep
                    if isInd:
                        independencies.append(spec)
                    if verbosity >= 4:
                        print('    independence test: ', spec, '=', isInd)
        
                        
        # Evaluate the independencies and resolve any conflicts.
        # A conflict occurs e.g.,  when A _!!_ B | C (implying C is a mediary for A--B), and
        # A _||_ C | B (implying B is a mediary for A--C).  These can't both be true,
        # so we choose the one with the lowest dependence.
        processed = {}
        for indep in independencies:
            v1, v2, conds = indep
            key = (v1, v2, tuple(conds))
            ivars = [v1, v2] + conds
            if key in processed:
                # We already handle this one as a conflict
                # with a previous.  Skip.
                continue
            conflicts = [key]
            for indep2 in independencies:
                if indep2 == indep:
                    continue
                v1_2, v2_2, conds_2 = indep2
                key2 = (v1_2, v2_2, tuple(conds_2))
                ivars_2 = [v1_2, v2_2] + conds_2
                conflict = True
                for ivar in ivars:
                    if ivar not in ivars_2:
                        # A conflict occurs if the same
                        # variables occur in two independencies.
                        conflict = False
                if not conflict:
                    continue
                # We have a conflict.
                conflicts.append(key2)
                processed[key2] = True
            if len(conflicts) > 1:
                # We had at least one conflict.  Resolve them.
                confdep = []
                for conflict in conflicts:
                    v1, v2, conds = conflict
                    try:
                        dep = ps.dependence(v1, v2, list(conds), power=power,
                                    sensitivity=sensitivity)
                    except:
                        assert False, 'checing dependence: ' + v1 + ',' + v2 + ', ' + str(list(conds))
                    confdep.append((dep, conflict))
                # Choose the one with the least dependence
                confdep.sort()
                winner = confdep[0][1]
                block = winner
                if verbosity >= 3:
                    print('      conflicts resolved: ', confdep)
            else:
                # No conflict.  This one is valid.
                block = key
            v1, v2, conds = block
            blocked[(v1, v2)] = conds
            if verbosity >= 2:
                if conds:
                    print('   ', v1, '--', v2, 'is blocked by', conds)
                else:
                    print('   ', v1, '--', v2, 'are independent')
            
            # Since we've identified a valid intermnediary relationship,
            # We can block relationships where an endpoint acts as a mediary to one
            # of our mediaries.
            for cond in conds:
                nmlist = nonmed.get((v1, cond), [])
                nmlist.append(v2)
                nonmed[(v1, cond)] = nmlist
                nmlist = nonmed.get((v2, cond), [])
                nmlist.append(v1)
                nonmed[(v2, cond)] = nmlist
    if verbosity >= 2:
        print('Detecting valid links.')
    # At this point, we have a dict of all invalid (blocked)
    # links.  Let's take all possible links, and remove the blocked.
    adjacencies = {}
    for v in varNames:
        adjacencies[v] = []
    ulinks = [] # Undirected Links (edges)
    dlinks = [] # Directed Links (edges)
    # Create list of undirected links, forming an undirected graph
    for i in range(len(varNames)):
        v1 = varNames[i]
        for j in range(i+1, len(varNames)):
            v2 = varNames[j]
            if (v1, v2) in blocked:
                continue
            ulinks.append((v1, v2))
            adjacencies[v1].append(v2)
            adjacencies[v2].append(v1)
    ugraph = nx.Graph(ulinks)
    maxPath = maxLevel + 2
    dirDict = {}
    for link in ulinks:
        # Now go through the undirected links, and direct them
        # to form a directed graph
        v1, v2 = link
        linkR = (v2, v1)
        if (ps.isCategorical(v1) or ps.cardinality(v1) == 2) and (ps.isCategorical(v2) or ps.cardinality(v2) == 2):
            # Both variables are either categorical or binary.  Use differential entropy
            de = dirDE(ps, v1, v2, verbosity=verbosity)
            vprint(4, verbosity, 'cdisc.discover: Differential Entropy for', link, 'is', de)
            dirDict[link] = de
            dirDict[linkR] = -de
        elif (ps.isCategorical(v1) or ps.cardinality(v1) == 2) or (ps.isCategorical(v2) or ps.cardinality(v2) == 2):
            # We have no method for this case
            dirDict[link] = 0
            dirDict[linkR] = 0
        else:
            adj1 = adjacencies[v1]
            adj2 = adjacencies[v2]
            allpaths = []
            #for path in nx.all_simple_paths(ugraph, v1, v2, maxPath):
            #    vprint(4, verbosity, 'cdisc.discover: found path for', link, '=', path)
            #        print('        cdisc.discover: found path for', link, '=', path)
            #    allpaths += path
            #pathvars = set(allpaths)
            #pathvars.remove(v1)
            #pathvars.remove(v2)
            #pathvars = list(pathvars)
            #vprint(4, verbosity'cdisc.discover: pathvars for', link, '=', pathvars)

            adj = list(set(adj1 + adj2))
            adj.remove(v1)
            adj.remove(v2)
            # adj now contains the set of adjacencies to either v1 or v2, without v1 or v2
            #dir = testDirection(ps, v1, v2, pathvars, power, maxLevel=maxLevel, verbosity=verbosity)
            dir = testDirection(ps, v1, v2, [], power, maxLevel=maxLevel, verbosity=verbosity)
            vprint(4, verbosity, 'cdisc.discover: Final direction for', link, 'is', dir)
            dirDict[link] = dir
            dirDict[linkR] = -dir
    # Now use triangular dependence to further weight the directions
    triangles = []
    for var in varNames:
        adj = adjacencies[var]
        if len(adj) < 2:
            # No triangles if not at least 2 adjacencies
            vprint(4, verbosity, 'No triangles for vertex = ', var)
            continue
        for i in range(len(adj)):
            a1 = adj[i]
            for j in range(i+1, len(adj)):
                a2 = adj[j]
                #if a2 in adjacencies[a1]:
                    # Eliminate triangles where the legs are directly connected.
                #    continue
                triangles.append((var,a1,a2))

    print('triangles = ', triangles)
    # First pass, we identify v-structures (dependence increasers) and pull links toward collider variables
    for triangle in triangles:
        var, a1, a2 = triangle
        l1 = (var, a1)
        l2 = (var, a2)
        l1R = (a1, var)
        l2R = (a2, var)
        #dep1 = ps.dependence(a1, a2, power=power, sensitivity=sensitivity)
        #dep2 = ps.dependence(a1, a2, [var],power=power, sensitivity=sensitivity)
        # If a1 and a2 are independent, but become dependent when conditioned on var, then
        # var is a collider, and we want to force the direction of both links toward var.
        isInd = ps.isIndependent(a1, a2 ,power=power, sensitivity=sensitivity)
        isIndC = ps.isIndependent(a1, a2, var ,power=power, sensitivity=sensitivity)
        vprint(4, verbosity, 'cdisc.discover.triangulation1: isInd, isIndC = ', isInd, isIndC)
        #condEff = dep2 - dep1
        #vprint(4, verbosity, 'cdisc.discover: Conditioning effect of', triangle[0], 'on', triangle[1:], '=', condEff)
        adjust = 0
        if isInd and not isIndC:
            adjust = -.3
        adjust1 = adjust2 = adjust


        if adjust1 != 0:
            dirDict[l1] += adjust
            dirDict[l1R] += -adjust
            vprint(4, verbosity, 'cdisc.discover.triangulation1', triangle, '.  Vertex is collider. Adjusting link', l1, 'by', adjust)
            dirDict[l2] += adjust
            dirDict[l2R] += -adjust
            vprint(4, verbosity, 'cdisc.discover.triangulation1', triangle, '.  Vertex is collider. Adjusting link', l2, 'by', adjust)
    # Second pass, we identify intermediaries (dependence reducers) and push the link with the most positive direction
    # away from the mediary
    for triangle in triangles:
        var, a1, a2 = triangle
        l1 = (var, a1)
        l2 = (var, a2)
        l1R = (a1, var)
        l2R = (a2, var)
        #dep1 = ps.dependence(a1, a2, power=power, sensitivity=sensitivity)
        #dep2 = ps.dependence(a1, a2, [var],power=power, sensitivity=sensitivity)
        isInd = ps.isIndependent(a1, a2, power=power, sensitivity=sensitivity)
        isIndC = ps.isIndependent(a1, a2, var, power=power, sensitivity=sensitivity)
        vprint(4, verbosity, 'cdisc.discover.triangulation2: isInd, isIndC = ', isInd, isIndC)

        #condEff = dep2 - dep1
        #vprint(4, verbosity, 'cdisc.discover: Conditioning effect of', triangle[0], 'on', triangle[1:], '=', condEff)
        adjust = 0
        l1dir = dirDict[l1]
        l2dir = dirDict[l2]
        if not isInd and isIndC:
            if l1dir <= 0 and l2dir <= 0:
                # This cannot be a collider.  Push the highest direction link away.
                if l1dir > l2dir:
                    # Push l1 away from var
                    adjust = .2
                    dirDict[l1] += adjust
                    dirDict[l1R] += -adjust
                    vprint(4, verbosity, 'cdisc.discover: triangulation2', triangle, '.  Vertex is medial.  Adjusting link', l1, 'by', adjust)
                else:
                    # Push l2 away from var
                    adjust = .2
                    dirDict[l2] += adjust
                    dirDict[l2R] += -adjust
                    vprint(4, verbosity, 'cdisc.discover: triangulation2', triangle, '.  Vertex is medial.  Adjusting link', l2, 'by', adjust)
            elif l1dir < 0 and l2dir == 0:
                # L1 incoming and L2 = 0.  Push L2 away
                adjust = .2
                dirDict[l2] += adjust
                dirDict[l2R] += -adjust
                vprint(4, verbosity, 'cdisc.discover: triangulation2', triangle, '.  Vertex is medial.  Adjusting link', l2, 'by', adjust)
            elif l2dir < 0 and l1dir == 0:
                # L2 incoming and L1 = 0.  Push L1 away
                adjust = .2
                dirDict[l1] += adjust
                dirDict[l1R] += -adjust
                vprint(4, verbosity, 'cdisc.discover: triangulation2', triangle, '.  Vertex is medial.  Adjusting link', l1, 'by', adjust)

    # Now process the directional information to form a (directed graph)
    for link in ulinks:
        v1, v2 = link
        dir = dirDict[link]
        if dir > 0:
            dlinks.append((v1, v2, dir))
        else:
            dir = -dir
            dlinks.append((v2, v1, dir))
        vprint(2, verbosity, 'cdisc.discover: found causal link:', dlinks[-1][0], '->', dlinks[-1][1], '(', dir, ')')

    # Resolve any loops in the graph due to improper direction detection.  This will form
    # a proper DAG.
    maxAttempts = 20
    resolved = False
    for attempt in range(maxAttempts):
        gr = nx.DiGraph()
        gr.add_nodes_from(varNames)
        lnks = [(link[0], link[1]) for link in dlinks]
        gr.add_edges_from(lnks)
        loops = list(nx.simple_cycles(gr))
        if len(loops) > 0:
            if attempt == 0:
                vprint(2, verbosity, 'Resolving Loops')
                vprint(2, verbosity, 'loops = ', loops)
            vprint(3, verbosity, 'links = ', dlinks)
            dlinks = resolveLoops(loops, dlinks, verbosity)
        else:
            if attempt > 0:
                vprint(2, verbosity, 'Loops resolved after', attempt, 'attempts.')
            resolved = True
            break
    if not resolved:
        vprint(1, verbosity, 'Failed to resolve loops.  Invalid model returned.', loops)
    # Now turn it into an RV model
    varDict = {}
    for var in varNames:
        varDict[var] = []
    for link in dlinks:
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
    # And from there, create a cGraph
    cg = cgraph.cGraph(gnodes, ps=ps, power=power, verbosity=verbosity)
    # Store the directional rho values for each edge, for later use.
    for link in dlinks:
        parent, child, rho = link
        cg.setEdgeProp((parent, child), 'dir_rho', rho)
    end = time.time()
    vprint(2, verbosity, 'cDisc.discover: Duration = ', round(end - start, 1))
    return cg


def autodiscover(ps, varNames=None, maxLevel=2, power=5, sensitivity=5, verbosity=2, minSensitivity=3, maxSensitivity=11, powers=[5], order=3):
    """
    Repeatedly discover and test Causal Model with various values for hyper-parameters (power and sensitivity).
    Return the best model along with the hyper-parameters used to generate it, and the final score for that model.
    Returns:
        (tuple) (cGraph instance, power, sensitivity, score)
    """
    bestSen = 0
    bestPow = 0
    bestScore = 0
    bestCG = None
    for p in powers:
        for s in range(minSensitivity, maxSensitivity+1):
            cg = discover(ps, varNames=varNames, maxLevel=maxLevel, power=p, sensitivity=s, verbosity=verbosity)
            score = cg.TestModel(order=order, power=p, sensitivity=s, testDirections=False, verbosity=verbosity)[0]
            if score > bestScore:
                if verbosity >= 2:
                    print('cdisc.autodiscover: got new interim best: power = ', p, ', sensitivity = ', s, ', score = ', score)
                bestSen = s
                bestPow = p
                bestScore = score
                bestCG = cg
    if verbosity >= 1:
        print('cdisc.autodiscover: Final best: power = ', p, ', sensitivity = ', s, ', score = ', score)
    return bestCG, bestPow, bestSen, bestScore
