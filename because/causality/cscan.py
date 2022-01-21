import sys
import time

from because.causality import RV
from because.causality import cGraph
from because.synth import read_data
from because.probability import independence

class Scanner:
    def __init__(self, testName):
        self.testName = testName
        f = open(testName, 'r')
        exec(f.read(), globals())
        self.gnodes = []
        # 'model' is set when the text file is exec'ed
        for var in model:
            observed = True
            dType = 'Numeric'
            if type(var) == type((0,)):
                name, parents = var[:2]
                if len(var) >= 3:
                    observed = var[2]
                if len(var) >= 4:
                    dType = var[3]
            else:
                name = var
                parents = []
            gnode = RV(name, parents, observed, dType, None, None)
            self.gnodes.append(gnode)

        # For dat file, use the input file name with the .csv extension
        tokens = testName.split('.')
        testFileRoot = str.join('.',tokens[:-1])
        datFileName = testFileRoot + '.csv'

        d = read_data.Reader(datFileName)
        self.data = d.read()

        self.g = cGraph(self.gnodes, self.data)
        self.clusters = {}
        self.exos = []

    def findParentClusters(self, cluster):
        clusterList = list(self.clusters.keys())
        potentialParents = []
        for c in clusterList:
            if len(c) >= len(cluster):
                continue
            potential = True
            for exo in c:
                if exo not in cluster:
                    potential = False
                    break
            if potential:
                potentialParents.append(c)
        return potentialParents

    def findParentVariables(self, cluster, parents=None):
        if parents is None:
            parentClusters = self.findParentClusters(cluster)
        else:
            parentClusters = parents
        vars = []
        for c in parentClusters:
            cvars = self.clusters[c]
            vars += cvars
        return set(vars)

    def getClusterExos(self, clustId):
        return list(clustId)

    def findBestParentLinks_orig(self, clustId, var=None):
        """Return links to the most likely parents from a given cluster to resolve each of its
            exogenous dependencies.
        """
        PWR = 1
        parentLinks = []
        # Get the list of possible parent clusters
        potentialParents = self.findParentClusters(clustId)
        print('Potential Parents for ', clustId, '=', potentialParents)
        cOrder = self.clusters[clustId]

        # If not otherwise specified, HeadVar is the first variable in the order of this cluster
        if var is None:
            headVar = cOrder[0]
        else:
            headVar = var
        # Find all the 
        exos = self.getClusterExos(clustId)
        if len(exos) == 1:
            # It's a top level cluster (only one exo)
            clustExo = clustId[0]
            for var in cOrder:
                if var == clustExo:
                    continue
                parentLinks.append((exos[0], var))
        else:
            # Not a top-level cluster.  We need to find the direct or
            # indirect link to each of its parents
            parents = []
            for exo in exos:
                # Check all potential indirect parents, and find the most likely
                bestMediator = None
                lowestDep = 1.0
                exoClusters = []
                for pc in potentialParents:
                    if exo not in self.getClusterExos(pc):
                        continue
                    exoClusters.append(pc)
                mediatorVars = self.findParentVariables(clustId, exoClusters)
                for m in mediatorVars:
                    if m == exo:
                        continue
                    dep = self.g.iProb.dependence(headVar, exo, m, raw=True, power=PWR)
                    print('dep(', headVar, exo, m, ') = ', dep)
                    if dep < lowestDep:
                        # The node that blocks the most dependence from headVar to exo
                        # is the best indirect parent.
                        lowestDep = dep
                        bestMediator = m
                if bestMediator is not None:
                    # Compare the best mediator with a direct connection and find the
                    # maximum dependence
                    medDep = self.g.iProb.dependence(headVar, bestMediator, raw=True)
                    dirDep = self.g.iProb.dependence(headVar, exo, raw=True)
                    if dirDep > medDep:
                        print('Using direct path for ', exo, headVar, dirDep, medDep)
                        parents.append(exo)
                    else:
                        parents.append(bestMediator)
                        print('Using indirect path for', exo, bestMediator, headVar, dirDep, medDep)
                else:
                    # No indirect path found.  Must be direct to the exo.
                    parents.append(exo)
            for parent in set(parents):
                # Use set to remove any duplicates
                parentLinks.append((parent, headVar))
        return parentLinks

    def findBestParentLinks(self, clustId, var=None):
        """Return links to the most likely parents from a given cluster to resolve each of its
            exogenous dependencies.
        """
        PWR = 1
        parentLinks = []
        # Get the list of possible parent clusters
        potentialParents = self.findParentClusters(clustId)
        print('Potential Parents for ', clustId, '=', potentialParents)
        cOrder = self.clusters[clustId]

        # If not otherwise specified, HeadVar is the first variable in the order of this cluster
        if var is None:
            headVar = cOrder[0]
        else:
            headVar = var
        # Find all the 
        exos = self.getClusterExos(clustId)
        if len(exos) == 1:
            # It's a top level cluster (only one exo)
            clustExo = clustId[0]
            for var in cOrder:
                if var == clustExo:
                    continue
                parentLinks.append((exos[0], var))
        else:
            # Not a top-level cluster.  We need to find the direct or
            # indirect link to each of its parents
            parents = []
            for exo in exos:
                # Check all potential indirect parents, and find the most likely
                bestParent = None
                highestDep = 0.0
                exoClusters = []
                for pc in potentialParents:
                    if exo not in self.getClusterExos(pc):
                        continue
                    exoClusters.append(pc)
                parentVars = self.findParentVariables(clustId, exoClusters)
                for p in parentVars:
                    dep = self.g.iProb.dependence(headVar, p, raw=True, power=PWR)
                    print('dep(', headVar, p, ') = ', dep)
                    if dep > highestDep:
                        # The node that blocks the most dependence from headVar to exo
                        # is the best indirect parent.
                        highestDep = dep
                        bestParent = p
                print('best parent for ', headVar, '->', exo, '=', bestParent)
                parents.append(bestParent)
            for parent in set(parents):
                # Use set to remove any duplicates
                parentLinks.append((parent, headVar))
        return parentLinks

    def Scan(self):
        g = self.g
        print('Testing: ', test, '--', testDescript)
        print()
        start = time.time()
        exos = g.findExogenous()
        self.exos = exos
        print()
        print('Exogenous variables = ', exos)
        # Map each exogenous variable to each non-exo that is dependent on it
        exoMap = g.findChildVars(exos)
        #print('ExoMap = \n', exoMap)
        print()
        # Transform ExoMap to get list of parent exos for each variable
        # exoMap2 := {varName1: [exoParent1, ...], varName2:[exoParent1, ...]}
        em2 = {}
        for t in exoMap:
            parent, child = t
            if child not in em2.keys():
                em2[child] = []
            if parent not in em2.keys():
                em2[parent] = []
            em2[child].append(parent)
        em3 = []
        for var in em2.keys():
            parents = em2[var]
            parents.sort()
            em3.append((len(parents), var, parents))
        em3.sort()
        em4 = [varSpec[1:] for varSpec in em3]

        print('Exogenous Map = \n', em4, '\n')
        print()
        # Convert ExoMap to a set of clusters with the same exo-parentage
        dType = 'Numeric'
        clusters = {}
        for var in exos:
            clusters[(var,)] = [var]
        for spec in em4:
            var, parents = spec
            if var in exos:
                continue
            clustKey = tuple(parents)
            if clustKey not in clusters.keys():
                clusters[clustKey] = []
            clusters[clustKey].append(var)
        self.clusters = clusters
        clustList = list(clusters.keys())
        print('clusters = ', clusters)
        print()
        # Find the approximate causal order of the variables within each cluster
        clusterLinks = []
        for clustId in clusters.keys():
            clustData = {}
            clustGNodes = []
            clustVars = clusters[clustId]
            memberCnt = 0
            # Populate a new cGraph with just the members of this cluster
            for var in clustVars:
                clustData[var] = self.data[var]
                clustGNodes.append(RV(var, [], True, dType, None, None))
                if var not in exos:
                    memberCnt += 1
            #if memberCnt > 1:
            if len(clustVars) > 1:
                #print('clustGNodes = ', [rv.name for rv in clustGNodes])
                cg = cGraph(clustGNodes, clustData)
                cOrder = cg.causalOrder()
                print('order for cluster', clustId, ' = ', cOrder)
                clusters[clustId] = cOrder
                for i  in  range(1, len(cOrder)):
                    testVar = cOrder[i]
                    parent = cOrder[i-1]
                    for exo in list(clustId):
                        exDep1 = g.iProb.dependence(parent, exo, raw=True)
                        exDep2 = g.iProb.dependence(testVar, exo, raw=True)
                        epsilon = .01
                        if exDep2 - exDep1 > epsilon:
                            print(testVar, 'contains more', exo, 'than ', parent, exDep1, exDep2, exDep2-exDep1, '.  This variable must be independently connected to Cluster = ', (exo,))
                            clusterLinks.append(((exo,), testVar))
                        #else:
                        #    print('testVar, parent, exo,  exdep1, exdep2, diff = ', testVar, parent, exo, exDep1, exDep2, exDep2 - exDep1)
            # Find the most likely links to this clusters parent(s)
            parentLinks = self.findBestParentLinks(clustId)
            clusterLinks += parentLinks
            clusterLinks2 = clusterLinks
            clusterLinks = []
            for link in clusterLinks2:
                parent, child = link
                if type(parent) == type((1,)):
                    # Parent is a cluster.  Try to resolve it.
                    cOrder = clusters[parent]
                    if len(cOrder) == 1:
                        clusterLinks.append((cOrder[0], child))
                    else:
                        # Add more resolution here.
                        clusterLinks.append(link)
                else:
                    # Resolved.
                    clusterLinks.append(link)
        # Resolve any remaining cluster -> node links
        print('clusterLinks = ', clusterLinks)
        clusterLinksR = []
        for link in clusterLinks:
            source, dest = link
            if type(source) == type(link):
                # Not fully resolved.  Let's try to resolve it.
                sourceMembers = clusters[source]
                bestParent = None
                mostDependence = 0
                targetExo = sourceMembers[0]
                for member in sourceMembers:
                    dep = g.iProb.dependence(member, dest, raw=True)
                    if dep > mostDependence:
                        bestParent = member
                        mostDependence = dep
                clusterLinksR.append((bestParent, dest))
            else:
                clusterLinksR.append(link)
        print()
        print('Resolved links = ', clusterLinksR)
        print()

        end = time.time()
        duration = end - start
        print('Test Time = ', round(duration))

if __name__ == '__main__':
    args = sys.argv
    if (len(args) > 1):
        test = args[1]
        s = Scanner(test)
        s.Scan()
