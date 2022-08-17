import sys
import time

from because.causality import RV
from because.causality import cGraph
from because.synth import read_data
from because.probability import independence

DEBUG = True
INDEPENDENCE_THRESHOLD = .75

class Scanner:
    def __init__(self, data={}, cg=None, power=1):
        self.power = power
        if cg is None:
            # No cgraph object passed in.  Build from data.
            gnodes = []
            self.varNames = []
            for var in data.keys():
                observed = True
                dType = 'Numeric'
                name = var
                self.varNames.append(var)
                parents = []
                gnode = RV(name, parents, observed, dType, None, None)
                gnodes.append(gnode)
            self.varNames.sort()
            self.g = cGraph(gnodes, data, power=self.power)
        else:
            # cgraph already exists
            self.g = cg
            self.varNames = self.g.varNames()
        self.rvList = self.g.rvList
        self.exos = []
        self.clustList = []
        self.clusterMembers = {}
        self.memberClusters = {}
        self.varLinks = []
        self.varGraph = {}
        self.clustGraph = {}

    def scan(self):
        """
        Scan the data to determine the hierarchical and peer causal relationships
        between variables.  Returns multiple results:
        - clustList -- A list of detected caual clusters
        - clustMembers -- A dictionary that maps: clusterName -> [memberVariableName, ...]
        - clustGraph -- A dictionary that maps: clusterName -> [parentClusterName, ...]
        - varGraph -- A dictionary that maps: variableName -> [parentVariableName, ...] 

        Returns:
            tuple: (clustList, clustMembers, clustGraph, varGraph) as described above.
        """
        g = self.g
        results = {}

        # Step 1 -- Find Exogenous Variables
        exos = self._findExogenous()
        # Set self.exos
        self.exos = exos
        if DEBUG: print('Exos = ', exos)
        results['exoVars'] = exos

        # Step 2 -- Identify clusters and their members
        clusterMembers, memberClusters = self._identifyClusterMembers()
        # Extract a sorted list of cluster names.
        # Set self.clustListm self,clusterMembers, self.memberClusters
        self.clustList = list(clusterMembers.keys())
        self.clustList.sort()
        self.clusterMembers = clusterMembers
        self.memberClusters = memberClusters
        if DEBUG:
            print('clustList = ', self.clustList)
            print('clustMembers = ', self.clusterMembers)
            print('memberClusters = ', self.memberClusters)

        # Step 3 -- Calculate Cluster Topologies
        clusterLinks = self._calculateClusterTopologies()
        if DEBUG: print('clusterLinks = ', clusterLinks)
        # Step 4 -- Resolve Inter-cluster Links
        resolvedLinks = self._resolveInterClustLinks(clusterLinks)
        if DEBUG: print('resolvedLinks = ', resolvedLinks)
        self.varLinks = resolvedLinks

        # Step 5 -- Convert Links to Graphs
        #   Convert links to a variable graph and a cluster graph
        #   Both graphs are DAGs
        #   Variable graph is map: variable -> list(parent variables)
        #   Cluster graph is map:  cluster -> list(parent cluster)
        clustGraph, varGraph = self._convertLinksToGraphs()
        self.clustGraph = clustGraph
        self.varGraph = varGraph
        # Emit all relevant results
        results['clusters'] = self.clustList
        results['clustMembers'] = self.clusterMembers
        results['clustGraph'] = self.clustGraph
        results['varGraph'] = self.varGraph
        return results

    def _findExogenous(self, include=None):
        """
        Find exogenous variables

        Args:
            exclude (list, optional): _description_. Defaults to [].
            power (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        if include is not None:
            rvList = include
        else:
            rvList = self.rvList
        if len(rvList) > 1:
            # Multiple variables.  Find the most exogenous
            rvList.sort()
            # First, use directional testing to accumulate "non-causality" for each variable
            accum = {}
            for v in rvList:
                accum[v] = 0.0
            numVars = len(rvList)
            for i in range(numVars):
                x = rvList[i]
                for j in range(i+1, numVars):
                    y = rvList[j]
                    R = self.g.testDirection(x, y)

                    if R > 0:
                        leastCausal = y
                    else:
                        leastCausal = x
                    accum[leastCausal] += abs(R)
            # Now sort them, and they will be approximately in order of most cauaal to least.
            scores = [(accum[key], key) for key in accum.keys()]
            scores.sort()
            exos = []
            for tup in scores:
                var = tup[1]
                if not exos:
                    # Always take the "most causal" variable to be exogenous
                    exos.append(var)
                else:
                    # Otherwise, test the variable for independence from all
                    # previous exogenous variables.
                    isExo = True
                    for exo in exos:
                        pval = self.g.testIndependence(var, exo, power=self.power)
                        if pval < INDEPENDENCE_THRESHOLD:
                            isExo = False
                            break
                            
                    if isExo:
                        exos.append(var)
                    else:
                        break
            exos.sort()
        else:
            # Only one variable.  Consider it exogenous.
            exos = rvList
        return exos

    def _identifyClusterMembers(self):
        """
        Identify the members of each cluster, and produce
        clusterMembers and memberClusters maps.

        Args:
            exoList (_type_): _description_
            power (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        # ClustMembers := map: clusterName -> [cluster members]
        clustMembers = {}
        # MemberClusters := map: clusterMember -> clusterName
        memberClusters = {}
        for var in self.rvList:
            pvars = []
            if var in self.exos:
                # Exos should go into the cluster with their name
                pvars.append(var)
            else:
                # Acccumulate any non independent exos for this variable
                # Its cluster becomes the set of exos on which it depends.
                for pvar in self.exos:
                    pval = self.g.testIndependence(pvar, var, power=self.power)
                    isInd = False
                    if pval > INDEPENDENCE_THRESHOLD:
                        isInd = True
                    print('ind ', pvar, '-', var, ' = ', pval)
                    if not isInd:
                        pvars.append(pvar)
            # Note: exoList is already sorted, so clustName will
            # be in canonical (alphabetical) order.
            if not pvars:
                continue
            clustName = tuple(pvars)
            if clustName not in clustMembers:
                clustMembers[clustName] = []
            clustMembers[clustName].append(var)
            memberClusters[var] = clustName
        return clustMembers, memberClusters

    def _causalOrder(self, clustId, members):
        # This algorithm needs improvement.
        maxTries = 10
        cOrder = []
        includes = members[:]
        # First, iteratively find the single most exogenous variable
        while len(cOrder) < len(members):
            exos = self._findExogenous(include=includes)
            cOrder += exos
            for exo in exos:
                includes.remove(exo)
        print('cluster', clustId, ' cOrder = ', cOrder)
        # Now try to find an order that is most parsimonious with the conditional probabilities
        for attempt in range(maxTries):
            correct = True
            for i in range(2, len(cOrder)):
                lowestDep = 2.0
                bestParent = None
                var1 = cOrder[i]
                var2 = cOrder[0]
                for j in range(len(cOrder)):
                    var3 = cOrder[j]
                    if var3 == var1 or var3 == var2:
                        continue
                    dep = self.g.testDependence(var1, var2, [var3], power=self.power)
                    #dep = self.iProb.dependence(var1, var2, var3, raw=True)
                    #print('dep', var1, var2, var3, '=', dep)
                    if dep < lowestDep:
                        lowestDep = dep
                        bestParent = var3
                if cOrder[i-1] != bestParent:
                    #print('Best parent for', var1, 'is', bestParent, '.  Current is', cOrder[i-1])
                    cOrder.remove(bestParent)
                    cOrder.insert(i, bestParent)
                    correct = False
                    break
            if correct:
                break
        if not correct:
            print('cscan.causalOrder: Could not converge to a definite order.')
        print('cluster', clustId, ' final order = ', cOrder)
        return cOrder

    def _calculateClusterTopologies(self):
        """
        Calculate the interior topology of each cluster, and the attachment
        points to parent clusters.
        Produces a set of fully or partially resolved inter-variable
        links.  Links are either:
        - (parentVarName, varName) or;
        - ([list parent exogenous vars], varName)

        Args:
            exos (_type_): _description_
            clusterMembers (_type_): _description_

        Returns:
            _type_: _description_
        """
        clusterLinks = []
        for clustId in self.clusterMembers.keys():
            clustData = {}
            clustGNodes = []
            headVars = []
            clustVars = self.clusterMembers[clustId]
            if len(clustVars) == 1:
                # Only one variable.  It must be where all the exos enter
                # the cluster.  Create partially resolved link.
                # Skip if top-level cluster (i.e. has no exos).
                if len(clustId) > 1:
                    clusterLinks.append((self._getClusterExos(clustId), clustVars[0]))
            else:
                # Find the approximate causal order of the variables within each cluster
                cOrder = self._causalOrder(clustId, clustVars)
                # Replace the clusterMemebers list with the detected order.
                self.clusterMembers[clustId] = cOrder
                # Add a partially resolved link between the list of all exos, and
                # our first variable.  The first variable must have all exos, or it
                # wouldn't be in this cluster.
                # Skip if top-level cluster (i.e. has no exos).
                if len(clustId) > 1:
                    clusterLinks.append((self._getClusterExos(clustId), cOrder[0]))
                for i  in  range(1, len(cOrder)):
                    testVar = cOrder[i]
                    parent = cOrder[i-1]
                    # Add a resolved link between this var and the previous in order
                    clusterLinks.append((parent, testVar))
                    # See if there are any redundant attachment points for exo parents.
                    redundExos = []
                    for exo in self._getClusterExos(clustId):
                        exDep1 = self.g.testDependence(parent, exo, power=self.power)
                        exDep2 = self.g.testDependence(testVar, exo, power=self.power)
                        epsilon = .05
                        print('testing for redundant links: ', testVar, exo, parent, exDep1, exDep2, exDep2 - exDep1)
                        if exDep2 - exDep1 > epsilon:
                            print(testVar, 'contains more', exo, 'than ', parent, exDep1, exDep2, exDep2-exDep1, '.  This variable must be independently connected to Cluster = ', (exo,))
                            # Redundant attachment point found. Add to redundExos list
                            redundExos.append(exo)
                    if redundExos:
                        # This variable has at least one redundant attachment.
                        # Create a partially resolved link for that.
                        if len(clustId) > 1:
                            clusterLinks.append((redundExos, testVar))
        return clusterLinks

    def _resolveInterClustLinks(self, inLinks):
        """
        Fully resolve links between clusters at a variable to
        variable level.
        Incoming links are either fully resolve (in which case
        we pass them through) or partially resolved, in which case
        we need to figure out which parent cluster(s) they attach to, 
        and where in those clusters they attach.

        Args:
            inLinks (_type_): _description_
        """
        resolved = []
        for link in inLinks:
            parent, child = link
            if type(parent) != type([]):
                # Link is fully resolved.  Pass it through.
                resolved.append(link)
            else:
                # Partially resolved.  Parent is a list of exos.
                parents = self._findBestParents(parent, child)
                for parent in parents:
                    resolved.append((parent, child))
        return resolved

    def _convertLinksToGraphs(self):
        """
        Convert a set of links to a variable graph and a cluster graph.
        Both graphs are DAGs.  Format is map: name -> [parent-name, ...]

        Args:
            allLinks (_type_): _description_
            memberClusters (_type_): _description_
            clustList (_type_): _description_
        """
        # Initialize empty graphs
        clustGraph0 = {}
        varGraph = {}
        # Add empty list of parents for each item in the graph
        for clust in self.clustList:
            clustGraph0[clust] = set()
        for var in self.varNames:
            varGraph[var] = []
        for link in self.varLinks:
            var1, var2 = link
            # Add this link info to the variable graph
            varGraph[var2].append(var1)
            # Get the clusters for both sides of the link
            clust1 = self.memberClusters[var1]
            clust2 = self.memberClusters[var2]
            if clust1 != clust2:
                # It's an inter-cluster link.  Add it to the
                # Cluster Graph
                clustGraph0[clust2].add(clust1)
        # Now convert sets to lists
        clustGraph = {}
        for clust in clustGraph0:
            clustSet = clustGraph0[clust]
            clustGraph[clust] = list(clustSet)
        return clustGraph, varGraph

    def _findPotentialParents(self, exoList):
        clusterList = self.clustList
        potentialParents = []
        for c in clusterList:
            if len(c) > len(exoList):
                # Lower level than any possible parent. Skip
                continue
            potential = True
            for exo in c:
                if exo not in exoList:
                    # This cluster has an exo not in our list.
                    #  Can't be the parent.
                    potential = False
                    break
            if potential:
                potentialParents.append(c)
        return potentialParents

    def _findParentVariables(self, cluster, parents=None):
        if parents is None:
            parentClusters = self._findParentClusters(cluster)
        else:
            parentClusters = parents
        vars = []
        for c in parentClusters:
            cvars = self.clusterMembers[c]
            vars += cvars
        return set(vars)

    def _getClusterExos(self, clustId):
        return list(clustId)

    def _findBestParents(self, parentExos, var):
        """Return the most likely parents for a given variable to resolve each of its
            exogenous dependencies.
        """
        parentLinks = []
        varCluster = self.memberClusters[var]
        # Get the list of possible parent clusters
        potentialParents = self._findPotentialParents(parentExos)
        print('Potential Parents for ', var, '=', potentialParents)
        potentialParentVars = []
        for potentialParent in potentialParents:
            # Make sure we don't allow resolution to the same cluster.
            if potentialParent != varCluster:
                potentialParentVars += self.clusterMembers[potentialParent]
        parentOrder = []
        for parentVar in potentialParentVars:
            if parentVar == var:
                continue
            # Check all potential parents, and find the most likely
            # Find the parent variable with the most dependence with our headVar
            dep = self.g.testDependence(parentVar, var, power=self.power)
            print('dep(', parentVar, var, ') = ', dep)
            parentOrder.append((dep, parentVar))
        parentOrder.sort()
        parentOrder.reverse()
        # Now we have a list of best attachment points in order of most to least
        # dependence.
        # We can only pick one parent from each cluster until we have satisfied
        # all or our exos.
        exosAccountedFor = set()
        bestParents = []
        for item in parentOrder:
            dep, parent = item
            parentClust = self.memberClusters[parent]
            # Make sure that this cluster adds new exos, and
            # isn't purely redundant
            isRedundant = True
            exos = self._getClusterExos(parentClust)
            for exo in exos:
                if exo not in exosAccountedFor:
                    exosAccountedFor.add(exo)
                    isRedundant = False
            if isRedundant:
                # This cluster is redundant.  Skip it.
                continue
            # We found another parent.  Add it, and see if
            # we've now accounted for all exos.
            bestParents.append(parent)
            if len(exosAccountedFor) == len(parentExos):
                # We've accounted for all our exos.  We're done.
                break
        print('best parents for ', parentExos, '->', var, '=', bestParents)
        return bestParents



