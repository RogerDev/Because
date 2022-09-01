
from because.causality import calc_indeps
from because.causality import rv
from because.causality import cgraph
import networkx as nx


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
                        cnt, dir, link = resDict[(var1, var2)]
                        cnt += 1
                    else:
                        cnt = 1
                    resDict[(var1, var2)] = (cnt, dir, link)
    resolutions = list(resDict.values())
    resolutions.sort()
    # Reverse the link with the most loops and smallest direction
    cnt, dir, rlink = resolutions[0]
    outlinks = []
    for link in links:
        if link == rlink:
            v1, v2, dir = link
            # Give it a high direction so it won't get reversed again.
            if verbosity >= 3:
                print('reversing link = ', link)
            outlinks.append((v2, v1, 1.0))
        else:
            outlinks.append(link)
    return outlinks

        
def discover(ps, varNames=None, maxLevel=2, power=5, sensitivity=5, verbosity=2):
    if varNames is None:
        varNames = ps.getVarNames()
    links = []
    
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
                        
        if verbosity >= 2:
            print('Evaluating conflicting interdependencies.')
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
                    dep = ps.dependence(v1, v2, list(conds), power=power,
                                sensitivity=sensitivity)
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
                mlist = nonmed.get((v2, cond), [])
                nmlist.append(v1)
                nonmed[(v2, cond)] = nmlist
    # At this point, we have a dict of all invalid (blocked)
    # links.  Let's take all possible links, and remove the blocked.
    for i in range(len(varNames)):
        v1 = varNames[i]
        for j in range(i+1, len(varNames)):
            v2 = varNames[j]
            if (v1, v2) in blocked:
                continue
            # Valid causal link.
            # Check the direction and add it.
            dir = ps.testDirection(v1, v2, power=power)
            if dir > 0:
                links.append((v1, v2, dir))
            else:
                dir = -dir
                links.append((v2, v1, dir))
            if verbosity >= 2:
                print('  found causal link:', links[-1][0], '->', links[-1][1], '(', dir, ')')
    # Resolve any loops in the graph due to improper direction detection.
    maxAttempts = 10
    for attempt in range(maxAttempts):
        gr = nx.DiGraph()
        gr.add_nodes_from(varNames)
        lnks = [(link[0], link[1]) for link in links]
        gr.add_edges_from(lnks)
        loops = list(nx.simple_cycles(gr))
        if len(loops) > 0:
            if attempt == 0:
                if verbosity >= 2:
                    print('Resolving Loops')

            if verbosity >= 2:
                print('  loops = ', loops)
            if verbosity >= 3:
                print('  links = ', links)
            links = resolveLoops(loops, links, verbosity)
        else:
            if verbosity >= 2 and attempt > 0:
                print('  Loops resolved after', attempt, 'attempts.')
            break
            
    # Now turn it into an RV model
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
    # And from there, create a cGraph
    cg = cgraph.cGraph(gnodes, ps=ps, power=power)
    return cg



    