"""
Causal Graph Module.

Implements class cGraph for defining and analyzing causal models.
"""
import math
from termios import IMAXBEL

import networkx as nx
import numpy as np

from because.probability import independence
from because.probability import ProbSpace
from because.probability import PDF
from because.probability import direction
from because.probability.standardiz import standardize
from because.visualization import grid2 as grid

DEBUG = False
# rvList is a list of random variable (rv) objects
# data is a dictionary keyed by random variable name, containing a list of observed values.
#   Each variable's list should be the same length
class cGraph:
    """
    Class for storing and operating on causal graphs.

    Causal graphs are stored as a directed acyclic graph (DAG).
    Uses networkx for graph analysis.

    The graph can be constructed with or without data, but many functions will not be useful
    without data.
    
    Args:
        rvList(list) List of Random Variable objects including their causal relationships (see rv.py).
                This represents the causal model.
        data(dictionary) Dictionary mapping each observed variable name to a list of observed values.
                    The value lists for all variables should be the same length (i.e. nObservations).
                    Defaults to {}.
        ps probability.prob.ProbSpace object.

        Must include rvList and either data or ps.  ProbSpace contains its own data.
    """
    def __init__(self, rvList, data={}, ps=None, power=None, verbosity=1):
        assert (data or ps) and not (data and ps), 'cGraph: Must pass either data or ps (ProbSpace object), but not both.'
        if power is None:
            if ps:
                self.power = ps.power
            else:
                self.power = power
        else:
            self.power = power
        self.verbosity = verbosity
        self.g = nx.DiGraph()
        self.rvDict = {}
        for rv in rvList:
            if rv.name in self.rvDict.keys():
                raise 'Duplicate variable name = ' + rv.name
            self.rvDict[rv.name] = rv
        self.rvList = list(self.rvDict.keys())
        self.rvList.sort()
        self.g.add_nodes_from(self.rvDict.keys())
        edges = []
        for rv in rvList:
            for pa in rv.parentNames:
                edges.append((pa, rv.name))
        self.g.add_edges_from(edges)
        self.data = data
        edges = self.g.edges()
        self.edgeDict = {}
        for edge in edges:
            s, d = edge
            if s in self.edgeDict.keys():
                self.edgeDict[s].append(edge)
            else:
                self.edgeDict[s] = [edge]
            if d in self.edgeDict.keys():
                self.edgeDict[d].append(edge)
            else:
                self.edgeDict[d] = [edge]
        self.edgeProps = {}
        # Create a probability space object for later use, if one wasn't passed in.
        if not ps:
            self.prob = ProbSpace(self.data, power=self.power)
        else:
            self.prob = ps
            power = ps.power
        # Create a separate standardized probability space for independence testing.
        iData = {}
        for var in self.data.keys():
            iData[var] = standardize(self.data[var])

        self.bdCache = {}
        self.fdCache = {}
        self.indepCache = {}
        self.dirCache = {}

    def getRVs(self):
        return [self.rvDict[varName] for varName in self.rvList]

    def setIndepCache(self, cache):
        """
        Set the Independence Cashe.
        Cache is a dictionary of {cacheKey: p-val}
        CacheKey is (source, dest, (condition-on))
        Source, dest are variable names
        condition-on is a tuple of variable names.
        Note, each source dest pair should be in the cache
        twice, once with source and dest reversed, since
        independence(A,B) = independence(B,A)

        Args:
            cache (dictionary): A cache dictionary as above.
        """
        self.indepCache = cache

    def setDirCache(self, cache):
        """
        Set the Direction cache.
        Cache is a dictionary of {cacheKey: rho}
        CacheKey is (source, dest).
        Source, dest are variable names.
        Note, each source dest pair should be in the cache
        twice, once with source and dest reversed with the
        value of rho negated, since dir(A, B) = -dir(B,A).

        Args:
            cache (dictionary): A Cache Dictionary as above.
        """
        self.dirCache = cache

    def varNames(self):
        """
        Get a list of all variable names in canonical order.

        Returns:
            list: List of variable name strings.
        """
        return self.rvList

    def vPrint(self, varName):
        """
        Print statistical information about a variable

        Args:
            varName (string): Variable name.
        """
        d = self.prob.distr(varName)
        print('stats(', varName,':', 'mean =', d.E(), ', stDev =', d.stDev(), ', skew =',  d.skew(), ', kurtosis =', d.kurtosis())

    def addLink(self, source, dest, verbosity=None):
        if verbosity is None:
            verbosity = self.verbosity
        assert source in self.rvDict, 'cGraph.addLink: Variable ' + source + ' not found. Link not added.'
        assert dest in self.rvDict, 'cGraph.addLink: Variable ' + dest + ' not found. Link not added.'
        rv1 = self.rvDict[source]
        rv2 = self.rvDict[dest]
        pList = rv2.parentNames
        if source in pList:
            if verbosity >= 1:
                print('cGraph.addLink: Link already exists (' + source + ', ' + dest + ').  No action taken.')
            return
        rv2.parentNames.append(source)
        self.g.add_edge(source, dest)
        self.edgeDict[source].append((source, dest))
        self.edgeDict[dest].append((source, dest))
        loops = list(nx.simple_cycles(self.g))
        if loops:
            print('cGraph.addLink: This link (' + source + ',' + dest + ') created a loop in the graph= ' + str(loops) + '.  Removing link.')
            self.removeLink(source, dest, verbosity=verbosity)
            return
        if self.verbosity >= 2:
            print('cGraph.addLink: Link ('+ source + ',' + dest + ') successfully added')

    def removeLink(self, source, dest, verbosity=None):
        if verbosity is None:
            verbosity = self.verbosity
        assert source in self.rvDict, 'cGraph.removeLink: Variable ' + source + ' not found. No change made.'
        assert dest in self.rvDict, 'cGraph.removeLink: Variable ' + dest + ' not found. No change made.'
        rv1 = self.rvDict[source]
        rv2 = self.rvDict[dest]
        pList = rv2.parentNames
        if source not in pList:
            if verbosity >= 1:
                print('cGraph.removeLink: Link doesn\'t exists (' + source + ', ' + dest + ').  No action taken.')
        rv2.parentNames.remove(source)
        self.g.remove_edge(source, dest)
        self.edgeDict[source].remove((source, dest))
        self.edgeDict[dest].remove((source, dest))
        if self.verbosity >= 2:
            print('cGraph.addLink: Link ('+ source + ',' + dest + ') successfully removed')

    def reverseLink(self, source, dest, verbosity=None):
        if verbosity is None:
            verbosity = self.verbosity
        assert source in self.rvDict, 'cGraph.reverseLink: Variable ' + source + ' not found. No change made.'
        assert dest in self.rvDict, 'cGraph.reverseLink: Variable ' + dest + ' not found. No change made.'
        rv1 = self.rvDict[source]
        rv2 = self.rvDict[dest]
        if rv1.name in rv2.parentNames:
            rv2.parentNames.remove(rv1.name)
            rv1.parentNames.append(rv2.name)
            self.g.remove_edge(source, dest)
            self.g.add_edge(dest, source)
            self.edgeDict[source].remove((source,dest))
            self.edgeDict[source].append((dest, source))
            self.edgeDict[dest].remove((source,dest))
            self.edgeDict[dest].append((dest, source))
            if verbosity >=2:
                print('cGraph.reverseLink: (', source, dest, ') has been changed to (', dest, source, ')')
        else:
            print('cGraph.reverseLink: Link(', source, dest, ') not found.  No changes made')

    def isExogenous(self, varName):
        """
        Return True if variable is exogenous (i.e. has no parents)

        Args:
            varName (string): Variable name.

        Returns:
            Boolean: True if exogenous, otherwise False.
        """
        rv = self.rvDict[varName]
        return not rv.parentNames

    def printGraph(self):
        """
        Prints the nodes and edges of the graph.
        """
        print('Nodes:', self.g.nodes())
        print('Edges:', self.g.edges())


    def getEdges(self):
        """
        Return a list of all edges in the graph

        Returns:
            list: list of tuples (fromNode, toNode)
        """
        return list(self.g.edges)

    def getEdgeProps(self, edge):
        props = self.edgeProps.get(edge, {})
        return props

    def getEdgeProp(self, edge, propName):
        props = self.getEdgeProps(edge)
        prop = props.get(propName, None)
        return prop

    def setEdgeProp(self, edge, propName, val):
        if edge in self.edgeProps:
            props = self.edgeProps[edge]
            props[propName] = val
        else:
            props = {propName: val}
            self.edgeProps[edge] = props

    def getAdjacencies(self, varName):
        """
        Returns links containing parents and children of a given variable

        Args:
            varName (string): Variable name.

        Returns:
            list: List of tuples: (parent, child) for any of the variables relationships.
        """
        if varName in self.edgeDict.keys():
            return self.edgeDict[varName]
        return []

    def getAdjacentVars(self, varName):
        """
        Returns parents and children of a given variable

        Args:
            varName (string): Variable name.

        Returns:
            list: List of variable names that are either parents or children
            of the given varName.
        """
        parents = self.getParents(varName)
        children = self.getChildren(varName)
        return parents + children

    def getParents(self, varName):
        """
        Return parents of the given variable

        Args:
            varName (string): Variable Name.

        Returns:
            list: List of parent names.
        """
        parents = []
        adj = self.getAdjacencies(varName)
        for a in adj:
            if a[1] == varName:
                parents.append(a[0])
        return parents

    def getChildren(self, varName):
        """
        Return children of the given variable

        Args:
            varName (string): Variable Name.

        Returns:
            list: List of child names.
        """
        children = []
        adj = self.getAdjacencies(varName)
        for a in adj:
            if a[0] == varName:
                children.append(a[1])
        return children

    def isChild(self, parentVar, childVar):
        """
        Returns True if parentVar is a parent of childVar.

        Args:
            parentVar (string): Parent variable name.
            childVar (string): Child variable name.

        Returns:
            Boolean: True if childVar is a child of parentVar, otherwise False.
        """
        if parentVar in self.getParents(childVar):
            return True
        return False

    def getAncestors(self, node):
        """
        Get all ancestors of a given node.

        Args:
            node (string): A given variable name.

        Returns:
            list: List of ancestor variable names
        """
        ancestors = set()
        parents = self.getParents(node)
        for parent in parents:
            ancestors.add(parent)
            ancs = self.getAncestors(parent)
            for anc in ancs:
                ancestors.add(anc)
        return list(ancestors)

    def isDescendant(self, ancestor, descendant):
        """
        Determine if a variable is a descendant of an ancestor variable.

        Args:
            ancestor (string): The ancestor variable name.
            descendant (string): The potential descendant variable name.

        Returns:
            bool: True if descendant is actually a descendant of ancestor
                    in the graph.  Otherwise False.
        """
        parents = self.getParents(descendant)
        if ancestor in parents:
            return True
        else:
            for parent in parents:
                 if self.isDescendant(ancestor, parent):
                    return True
        return False

    def isAdjacent(self, node1, node2):
        """
        Determine if two variable are adjacent in the causal model graph.

        Args:
            node1 (string): First variable name.
            node2 (string): Second variable name.

        Returns:
            bool: True if the first and second variables are adjacent.
                        Otherwise False.
        """
        adj = self.getAdjacencies(node1)
        for a in adj:
            if a[0] == node2 or a[1] == node2:
                return True
        return False

    def combinationsxxx(self, inSet):
        """
        Find all combinations of a set of variable names.

        Args:
            inSet (set): A set of variable names.

        Returns:
            list: A list of all combinations of the variables in the inSet.
        """
        c = []
        
        for i in range(len(inSet)):
            u = inSet[i]
            for v in inSet[i+1:]:
                c.append((u, v))
        return c

    def makeDependency(self, u, v, w, isDep):
        """
        Given a set of three variable names (u,v,w) and
        an expectation of dependency, return a canonical
        dependency tuple.  Makes sure names are in canonical
        (alphabetical, order).

        Args:
            u (string): First variable
            v (_type_): Second variable
            w (_type_): Third (conditional) variable
            isDep (bool): True if the variables are expected to be
                    conditionaly dependent 

        Returns:
            tuple: 4-tuple --  (var, var, w, isDep)
        """
        if u < v:
            d = (u, v, w, isDep)
        else:
            d = (v, u, w, isDep)
        return d

    def getCombinations(self, nodes=None, order=3, minOrder = 1):
        #print ('order = ', order)
        from itertools import combinations
        if nodes is None:
            nodes = self.g.nodes()
        allCombos = []
        for o in range(minOrder, order+1):
            combos = combinations(nodes, o)
            allCombos += combos
        #print('allCombos = ', allCombos)
        return allCombos


    def computeDependencies(self, order):
        deps = []
        nodes = list(self.g.nodes())
        nodes.sort()
        #print('nodes = ', nodes, ', order = ', order)
        cNodes = self.getCombinations(nodes, order)
        for i in range(len(nodes)):
            node1 = nodes[i]
            if not self.rvDict[node1].isObserved:
                continue
            for j in range(i, len(nodes)):
                node2 = nodes[j]
                if node1 == node2 or not self.rvDict[node2].isObserved:
                    continue
                isAdjacent = self.isAdjacent(node1, node2)
                isSeparated = not isAdjacent and nx.d_separated(self.g, {node1}, {node2}, {})
                dep = self.makeDependency(node1, node2, None, not isSeparated)
                deps.append(dep)
                for c in cNodes:
                    #print('cNodes = ', cNodes)
                    if node1 in c or node2 in c:
                        continue
                    # Verify that every member of c is observed.  If not, we skip this combo.
                    allObserved = True
                    for m in c:
                        if not self.rvDict[m].isObserved:
                            allObserved = False
                            break
                    if not allObserved:
                        continue
                    isSeparated = not isAdjacent and nx.d_separated(self.g, {node1}, {node2}, set(c))
                    dep = self.makeDependency(node1, node2, list(c), not isSeparated)
                    deps.append(dep)
        #print('deps = ', deps)
        return deps
    
    def formatDependency(self, dep):
        # dep is:  from, to, given, isDependent
        u, v, w, isDep = dep
        if isDep:
            rel = 'is not independent from'
        else:
            rel = 'is independent from'
        if w is None:
            given = ''
        else:
            given = 'given ' + str(w)
        out = u + ' ' + rel + ' ' + v + ' ' + given
        return out

    def printDependencies(self, deps):
        print('Implied Dependencies:\n')
        for d in deps:
            print(self.formatDependency(d))

    def testIndependence(self, x, y, z=[], power=1, sensitivity=5):
        """
        Test the independence between two variables, with zero or more
        conditional variables.
        Note: If A and B are independent then so are B and A.

        Args:
            x (string): The first variable name.
            y (string): The second variable name.
            z (list): A list of zero or more variables on which to condition.
                        Optional, Default=[].

        Returns:
            string: P-val -- A value between 0 and 1.  Values > .5 imply independence.
                        Values < .5 imply dependence.  Values near .5 are ambiguous.
        """
        cacheKey = (x, y, tuple(z))
        #print('testing independence', cacheKey)
        if cacheKey in self.indepCache:
            independence = self.indepCache[cacheKey]
        else:
            dependence = self.prob.dependence(x, y, z, power=power, sensitivity=sensitivity)
            independence = 1 - dependence
            self.indepCache[cacheKey] = independence
            reverseKey = (y, x, tuple(z))
            self.indepCache[reverseKey] = independence
        return independence

    def testDependence(self, x, y, z=[], power=1, sensitivity=5):
        """
        Test the dependence between two variables, with zero or more
        conditional variables.
        Note: If A and B are dependent then so are B and A.

        Args:
            x (string): The first variable name.
            y (string): The second variable name.
            z (list): A list of zero or more variables on which to condition.
                        Optional, Default=[].

        Returns:
            string: P-val -- A value between 0 and 1.  Values < .5 imply independence.
                        Values > .5 imply dependence.  Values near .5 are ambiguous.
        """
        return 1 - self.testIndependence(x, y, z, power=power, sensitivity=sensitivity)


    def TestModel(self, data=None, order=3, power=None, sensitivity=5, deps=None, edges=None, testDirections=True, verbosity=None):
        """
        Test the model for consistency with a set of data.
        Format for data is {variableName: [variable value]}.
        That is, a dictionary keyed by variable name, containing
        a list of data values for that variable's series.
        The lengths of all variable's lists should match.
        That is, the number of samples for each variable must
        be the same.

        Returns: 
            tuple: (confidence, numTotalTests, [numTestsPerType], [numErrsPerType], [errorDetails])
        Where:
            - confidence is an estimate of the likelihood that the data generating process defined
            by the model produced the data being tested.  Ranges from 0.0 to 1.0.
            - numTotalTests is the number of independencies and dependencies implied by the model.
            - numTestsPerType is a list, for each error type, 0 - nTypes, of the number of tests that
            test for the given error type.
            - numErrsPerType is a list, for each error type, of the number of failed tests.
            - errorDetails is a list of failed tests, each with the following format:
                [(errType, x, y, z, isDep, pval, errStr)]
                Where:
                    errType = 
                        0 (Exogenous variables not independent) or;
                        1 (Expected independence not observed) or; 
                        2 (Expected dependence not observed) or;
                        3 (Incorrect Direction)
                    x, y, z are each a list of variable names that
                        comprise the statement x _||_ y | z.
                        That is x is independent of y given z.
                    isDep True if a dependence is expected.  False for 
                        independence
                    pval -- The p-val returned from the independence test
                    errStr A human readable error string describing the error
            - warningDetails is a list of tests with warnings.  Format is the same as
                for errorDetails above.
        """
        if power is None:
            power = self.power
        if verbosity is None:
            verbosity = self.verbosity
        warningPenalty = .0025
        ps = self.prob
        numTestTypes = 4
        errors = []
        warnings = []
        if data is None:
            data = self.data
        numTestsPerType = [0] * numTestTypes
        numErrsPerType = [0] * numTestTypes
        numWarnsPerType = [0] * numTestTypes
        if deps is None:
            # No dependencies passed in.  Compute them here.
            deps = self.computeDependencies(order)
        if verbosity >= 2:
            print('Testing Model for', len(deps), 'Independencies')
        for dep in deps:
            x, y, z, isDep = dep
            if z is None:
                z = []
            #pval = independence.test(ps, [x], [y], z, power=power)
            pval = self.testIndependence(x, y, z, power=power, sensitivity=sensitivity)
            if verbosity >= 4:
                print('   Testing', x, y, z)
            errStr = None
            warnStr = None
            testType = -1
            if not z and self.isExogenous(x) and self.isExogenous(y):
                testType = 0
            elif not isDep:
                testType = 1
            else:
                testType = 2
            if testType == 2:
                pval = self.testIndependence(x, y, z, power=power, sensitivity=sensitivity)
            else:
                pval = self.testIndependence(x, y, z, power=power, sensitivity=sensitivity)

            numTestsPerType[testType] += 1
            if testType == 0 and pval < .5:
                    errStr = 'Error (Type 0 -- Exogenous variables not independent) -- Expected: ' + self.formatDependency(dep) + ' but dependence was detected. P-val = ' + str(pval)
            elif testType == 2 and pval > .5:
                if pval > .75:
                    errStr = 'Error (Type 2 -- Unexpected independence) -- Expected: ' +  self.formatDependency(dep) + ' but no dependence detected.  P-val = ' + str(pval)
                else:
                    warnStr = 'Warning (Type 2 -- Unexpected independence) -- Expected: ' +  self.formatDependency(dep) + ' but minimal dependence detected.  P-val = ' + str(pval)
            elif testType == 1 and pval < .5:
                if pval < .25:
                    errStr = 'Error (Type 1 -- Unexpected dependence) -- Expected: ' + self.formatDependency(dep) + ' but dependence was detected. P-val = ' + str(pval)
                else:
                    warnStr = 'Warning (Type 1 -- Unexpected dependence) -- Expected: ' + self.formatDependency(dep) + ' but some dependence was detected. P-val = ' + str(pval)
            if errStr:
                if verbosity >= 3:
                    print('   ***', errStr)
                errors.append((testType, [x], [y], list(z), isDep, pval, errStr))
                numErrsPerType[testType] += 1
            if warnStr:
                if verbosity >= 3:
                    print('   *', warnStr)
                warnings.append((testType, [x], [y], list(z), isDep, pval, warnStr))
                numWarnsPerType[testType] += 1
            elif verbosity >= 4:
                print('   .',)
        # Now test directionalities
        dresults = []
        if testDirections:
            testType = 3
            dresults = self.testAllDirections(edges = edges, power=power)
            derrs = 0
            for dresult in dresults:
                isError, cause, effect, rho = dresult
                if isError:
                    derrs += 1
                    if abs(rho) < .0001:
                        resStr = 'True direction could not be verified.'
                        warnStr = 'Warning (Type 3 -- Incorrect Causal Direction) between ' + cause + ' and ' + effect + '. ' + resStr + '.  Rho = ' + str(rho)
                        warnings.append((testType, [cause], [effect], [], False, rho, warnStr))
                        if verbosity >= 3:
                            print('   *', warnStr)
                    else:
                        resStr = 'Direction appears to be reversed.'
                        errStr = 'Error (Type 3 -- Incorrect Causal Direction) between ' + cause + ' and ' + effect + '. ' + resStr + '.  Rho = ' + str(rho)
                        errors.append((testType, [cause], [effect], [], False, rho, errStr))
                        if verbosity >= 3:
                            print('   ***', errStr)
                        numErrsPerType[testType] += 1
                numTestsPerType[testType] += 1
        confidence = self.scoreModel(numTestsPerType, numErrsPerType, numWarnsPerType)
        numTotalTests = len(deps) + len(dresults)
        if verbosity >= 2:
            print('Model Testing Completed with', len(errors), 'error(s) and', len(warnings), ' warning(s).  Confidence = ', round(confidence * 100, 1), '%')
        return (confidence, numTotalTests, numTestsPerType, numErrsPerType, numWarnsPerType, errors, warnings)

    def scoreModel(self, numTestsPerType, numErrsPerType, numWarnsPerType=[]):
        """
        Score the confidence in the model base on results of the 4 test types:
        - 0 -> Exogenous Variable not Independent
        - 1 -> Unexpected Dependence
        - 2 -> Unexpected Independence
        - 3 -> Incorrect Causal Direction

        Args:
            numTestsPerType (sequence): Sequence of 4 integers representing the count of tests of each of
                the four types (0-3).
            numErrsPerType (sequence): Sequence of 4 integers representing the count of failed test for
                each of the four types
            numWarnsPerType (sequence): Sequence of 4 integers representing the number of tests with
                warnings for each of the four types.

        Returns:
            Integer: A confidence score from 0 (no correspondance) to 1.0 (perfect correspondence).
        """
        numTestTypes = 4
        failurePenaltyPerType = [1, 1, 1, 1] # Adjust to weight error types differently
        warningPenaltyPerType = [.25, .25, .25, .25] # Adjust to weight warning types differently

        confidence = 1.0
        for i in range(numTestTypes):
            nTests = numTestsPerType[i]
            nErrs = numErrsPerType[i]
            nWarns = numWarnsPerType[i]
            if nTests > 0:
                eratio = nErrs / nTests
                confidence -= eratio * failurePenaltyPerType[i] / numTestTypes
                wratio = nWarns / nTests
                confidence -= wratio * warningPenaltyPerType[i] / numTestTypes
        confidence = max([confidence, 0.0])
        return confidence


    def testDirection(self, x, y, power=None):
        """
        Test the implied directionality between two variables.
        rho > 0 implies forward direction (i.e. x -> y).
        rho < 0 implies reverse direction (i.e. Y -> x).
        rho ~= 0 means direction cannot be determined.

        Args:
            x (string): the source variable name
            y (string): the destination variable name

        Returns:
            float: rho as described above.
        """
        if power is None:
            power = self.power
        cacheKey = (x,y)
        if cacheKey in self.dirCache:
            rho = self.dirCache[cacheKey]
        else:
            #rho = direction.test_direction(self.data[x], self.data[y])
            # Use standardized data
            rho = self.prob.testDirection(x, y, power)
            # Add result to cache
            self.dirCache[cacheKey] = rho
            # Add reverse result to cache, with reversed rho
            #reverseKey = (y,x)
            #self.dirCache[reverseKey] = -rho
        return rho

    def testAllDirections(self, edges=None, power=None):
        if power is None:
            power = self.power

        epsilon = .0001
        if edges is None:
            edges = self.getEdges()
        results = []
        errors = 0
        for edge in edges:
            x, y = edge
            rho = self.testDirection(x, y, power)
            if rho > epsilon:
                isError = False
            else:
                isError = True
                errors += 1
            #print((isError, x, y, rho))
            results.append((isError, x, y, rho))
        return results

    def causalOrder(self, power=None, sensitivity=5):
        if power is None:
            power = self.power
        maxTries = 10
        cOrder = []
        correct = True
        while len(cOrder) < len(self.rvList):
            exos = self.findExogenous(exclude=cOrder, sensitivity=sensitivity)
            cOrder += exos
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
                    dep = self.testDependence(var1, var2, [var3], power=power, sensitivity=sensitivity)
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
            print('cGraph.causalOrder: Could not converge to a definite order.')
        return cOrder

    def findExogenous(self, exclude=[], power=None, N_train=100000, sensitivity=5):
        if power is None:
            power = self.power

        rvList = self.rvList
        rvList.sort()
        accum = {}
        for v in rvList:
            if v not in exclude:
                accum[v] = 0.0
        numVars = len(rvList)
        for i in range(numVars):
            x = rvList[i]
            if x in exclude:
                continue
            for j in range(i+1, numVars):
                y = rvList[j]
                if x == y or y in exclude:
                    continue
                R = self.testDirection(x, y, power, N_train, power=power)

                if R > 0:
                    leastCausal = y
                else:
                    leastCausal = x
                accum[leastCausal] += abs(R)
        scores = [(accum[key], key) for key in accum.keys()]
        scores.sort()
        exos = []
        for tup in scores:
            var = tup[1]
            if not exos:
                exos.append(var)
            else:
                isExo = True
                for exo in exos:
                    pval = self.testIndependence(var, exo, power=power, sensitivity=sensitivity)
                    #print('ind ', var, '-', exo, ' = ', pval)
                    if pval < .5:
                        isExo = False
                        break
                        
                if isExo:
                    exos.append(var)
                else:
                    break
        return exos

    def findChildVars(self, parentList, power=1, sensitivity=5):
        outMap = []
        for var in self.rvList:
            for pvar in parentList:
                if var in parentList:
                    continue
                pval = self.testIndependence(pvar, var, power=power, sensitivity=sensitivity)
                isInd = False
                if pval > .5:
                    isInd = True
                print('ind ', pvar, '-', var, ' = ', pval)
                if isInd:
                    continue
                outMap.append((pvar, var))
        return outMap

    def intervene(self, targetRV, doList, controlFor = [], power=None, verbosity=None):
        """ 
        Implements Intverventions (Level2 of Ladder of Causality)
        of the form P(Y | do(X1=x1),Z).  That is, the Probability
        of Y given that we set X1 to x1 and control for Z.  This is generalized
        to allow multiple interventions on different variables.
        doList is the set of interventions: [(varName1, val1), ..., (varNamek, valk)].
        We return a probability distribution that can be further queried,
        e.g., as to the probability of a value, or the expected value
        (see Probability/Prob.py and pdf.py)
        """
        if power is None:
            power = self.power
        if verbosity is None:
            verbosity = self.verbosity
        if verbosity >= 3:
            print('cGraph.intervene: target, doList, controlFor = ', targetRV, doList, controlFor)
        # Filter out any interventions for which the target is not a descendant of the
        # intevening variable.  The effect of those interventions will alsways be zero.
        doListF = []
        for item in doList:
            rv, value = item
            if targetRV in nx.descendants(self.g, rv):
                # It is a descendant.  Keep it.
                doListF.append(item)
        if not doListF:
            # No causal effects.  Return P(target)
            return self.prob.distr(targetRV, controlFor)

        # Find all the backdoor paths and identify the minimum set of variables (Z) that
        # block all such paths without opening any new paths.
        blockingSet = self.findBackdoorBlockingSet(doListF[0][0], targetRV)
        if verbosity >= 3:
            print('cGraph.intervene: backward blocking set = ', blockingSet)
        # Make sure that none of our intervention variables is in the blocking set.
        for item in doList:
            if item[0] in blockingSet:
                blockingSet.remove(item[0])
        # Now we compute the probability distribution of Y conditionalized on all of the blocking
        # variables.
        given = doList + blockingSet + controlFor
        if verbosity >= 3:
            print('cGraph.intervene: ProbSpace query = distr(', targetRV, given, ')')
        distr = self.prob.distr(targetRV, given, power=power)
        # We return the probability distribution
        return distr

    def ACE(self, cause, effect, power=None):
        """
        Average Causal Effect of cause on effect.
        """
        if power is None:
            power = self.power
        causeDistr = self.prob.distr(cause)
        causeMean = causeDistr.E()
        causeStd = causeDistr.stDev()
        tests = [.2, .5, 1 ]
        testResults = []
        for test in tests:
            lowBound = causeMean - (causeStd * test)
            highBound = causeMean + (causeStd * test)
            diff = highBound - lowBound
            effectAtLow = self.intervene(effect, [(cause, lowBound)], power=power).E()
            effectAtHigh = self.intervene(effect, [(cause, highBound)], power=power).E()
            ace = (effectAtHigh - effectAtLow) / diff
            testResults.append(ace)
        #print('testResults = ', testResults)
        tr = np.array(testResults)
        final = float(np.mean(tr))
        #print('ACE = ', effectAtMean, effectAtUpper, ace)
        return final


    def CDE(self, cause, effect, power=None):
        """
        Controlled Direct Effect of cause on effect
        """
        if power is None:
            power = self.power
        causeDistr = self.prob.distr(cause)
        causeMean = causeDistr.E()
        causeStd = causeDistr.stDev()
        if not self.isChild(cause, effect):
            # Can't have a direct effect if cause is not a parent of effect
            return 0.0
        bdBlocking = self.findBackdoorBlockingSet(cause, effect)
        fdBlocking = self.findFrontdoorBlockingSet(cause, effect)
        given = bdBlocking + fdBlocking
        tests = [.2, .5, 1 ]
        testResults = []
        for test in tests:
            lowBound = causeMean - (causeStd * test)
            highBound = causeMean + (causeStd * test)
            diff = highBound - lowBound
            effectAtLow = self.prob.distr(effect, [(cause, lowBound)] + given, power=power).E()
            effectAtHigh = self.prob.distr(effect, [(cause, highBound)] + given, power=power).E()
            cde = (effectAtHigh - effectAtLow) / diff
            testResults.append(cde)
        #print('testResults = ', testResults)
        tr = np.array(testResults)
        final = float(np.mean(tr))
        return final

    def MCE(self, cause, effect, power=None, verbosity=None):
        """
        Maximum Causal Effect.
        Returns the Maximum difference in the effect variable for all values of cause.
        This works for categorical as well as numeric variables.
        This includes direct and indirect relationships between cause and effect.
        For only direct relationships, see MDE.
        """
        if power is None:
            power = self.power
        if verbosity is None:
            verbosity = self.verbosity

        # Get the blocking set for the forward causal path
        fdBlocking = self.findFrontdoorBlockingSet(cause, effect)
        if verbosity >= 3:
            print('cGraph.MCE: forward blocking set =', fdBlocking)
        g = grid.Grid(self.prob, [cause], numPts=10)
        testpoints = g.makeGrid()
        numPoints = g.getTestCount()
        minv = 1000000000
        maxv = -1000000000
        iMin = None
        iMax = None
        i = 0
        for testspec in testpoints:
            testpoint = testspec[0]
            if len(testpoint) == 2:
                nom, val = testpoint
                effDistr = self.intervene(effect, [(cause, val)], fdBlocking, power=power, verbosity=verbosity)
                effVal = effDistr.E()
            else:
                nom, lowval, highval = testpoint
                if lowval:
                    val = lowval
                else:
                    val = highval
                effDistr = self.intervene(effect, [(cause, val)], fdBlocking, power=power, verbosity=verbosity)
            if effDistr is None:
                continue
            effVal = effDistr.E()
            if verbosity >= 3:
                print('cGraph.MCE: effVal for causeVal ', val, '=', effVal)
            if effVal < minv:
                minv = effVal
                iMin = i
            if effVal >= maxv:
                maxv = effVal
                iMax = i
            i += 1        
        effDistr = self.prob.distr(effect)
        # Use 99p - 1p as the effective range, unless discrete
        if self.prob.isDiscrete(effect):
            efflow = effDistr.minVal()
            effhigh = effDistr.maxVal()
        else:
            efflow = effDistr.percentile(1)
        effhigh = effDistr.percentile(99)
        effRange = effhigh - efflow
        cRange = maxv - minv
        if effRange > 0:
            mce = cRange / effRange
        elif cRange > 0:
            mce = 1
        else:
            mce = 0
        # Bound at 1.0.  It is possible to be over 1 due to the inexactness of the denominator.
        mce = min([mce, 1.0])
        #print('iMin, iMax, minv, maxv, mde = ',iMin, iMax, minv, maxv, mde)
        return mce

    def MDE(self, cause, effect, power=None, verbosity=None):
        """
        Maximum Direct Causal Effect (controlled).
        Returns the Maximum difference in the effect variable for all values of cause,
        when all indirect paths are blocked
        This works for categorical as well as numeric variables.
        This includes direct and indirect relationships between cause and effect.
        For only direct relationships, see MDE.
        """
        if power is None:
            power = self.power
        if verbosity is None:
            verbosity = self.verbosity
        g = grid.Grid(self.prob, [cause], numPts=10)
        testpoints = g.makeGrid()
        numPoints = g.getTestCount()
        minv = 1000000000
        maxv = -1000000000
        iMin = None
        iMax = None
        i = 0
        for testspec in testpoints:
            testpoint = testspec[0]
            if len(testpoint) == 2:
                nom, val = testpoint
                effDistr = self.intervene(effect, [(cause, val)], power=power, verbosity=verbosity)
                effVal = effDistr.E()
                    
            else:
                nom, lowval, highval = testpoint
                if lowval:
                    val = lowval
                else:
                    val = highval
                effDistr = self.intervene(effect, [(cause, val)], power=power, verbosity=verbosity)
            if effDistr is None:
                continue
            effVal = effDistr.E()
            if verbosity >= 3:
                print('cGraph.MDE: effVal for causeVal', val, '=', effVal)
            if effVal < minv:
                minv = effVal
                iMin = i
            if effVal >= maxv:
                maxv = effVal
                iMax = i
            i += 1        
        effDistr = self.prob.distr(effect)
        # Use 99p - 1p as the effective range, unless discrete
        if self.prob.isDiscrete(effect):
            efflow = effDistr.minVal()
            effhigh = effDistr.maxVal()
        else:
            efflow = effDistr.percentile(1)
        effhigh = effDistr.percentile(99)
        effRange = effhigh - efflow
        cRange = maxv - minv
        if effRange > 0:
            mde = cRange / effRange
        elif cRange > 0:
            mde = 1
        else:
            mde = 0
        # Bound at 1.0.  It is possible to be over 1 due to the inexactness of the denominator.
        mde = min([mde, 1.0])
        #print('iMin, iMax, minv, maxv, mde = ',iMin, iMax, minv, maxv, mde)
        return mde

    def findBackdoorBlockingSet(self, source, target, verbosity=None):
        """ 
        Find the minimal set of nodes that block all backdoor paths from source
        to target.
        """
        cacheKey = (source, target)
        if cacheKey in self.bdCache.keys():
            return self.bdCache[cacheKey]
        if verbosity is None:
            verbosity = self.verbosity
        maxBlocking = 3
        bSet = []
        # find all paths from parents (including ancestors) of source to target.
        parents = self.getParents(source)
        if verbosity >= 3:          
            print('findBackdoorBloockingSet: parents = ', parents)
        # Create a graph view that removes the links from the source to its parents
        def includeEdge(s, d):
            #print('source, dest = ', s, d)
            if d == source:
                return False
            return True
        pathNodes = {}
        vg = nx.subgraph_view(self.g, filter_edge=includeEdge)
        for parent in parents:
            paths = nx.all_simple_paths(vg, parent, target)
            for path in paths:
                if verbosity >= 3:
                    print('findBackdoorBloockingSet: path = ', path)
                # Remove the last node of the path -- always the target
                intermediates = path[:-1]
                if verbosity >= 3:
                    print('findBackdoorBloockingSet: int = ', intermediates)
                for i in intermediates:
                    if i not in pathNodes:
                        pathNodes[i] = 1
                    else:
                        pathNodes[i] += 1
        # Remove any descendants of source from pathNodes
        allNodes = list(pathNodes.keys())
        for node in allNodes:
            if self.isDescendant(source, node):
                del pathNodes[node]
        pathTups = []
        # First look for single node solutions
        for node in pathNodes.keys():
            cnt = pathNodes[node]
            outTup = (cnt, node)
            pathTups.append(outTup)
               
        # Sort the nodes in descending order of the number of paths containing it
        pathTups.sort()
        pathTups.reverse()
        combos = [(tup[1],) for tup in pathTups]
        if verbosity >= 3:
            print('findBackdoorBloockingSet: pathNodes = ', pathNodes.keys())
        # Now add any multiple field combinations.  Order is not significant here.
        multiCombos = self.getCombinations(pathNodes.keys(), maxBlocking, minOrder=2)
        combos += multiCombos
        if verbosity >= 3:
            print('findBackdoorBloockingSet: combos = ', combos)
        for nodeSet in combos:
            testSet = set(nodeSet)
            if verbosity >= 3:
                print('findBackdoorBloockingSet: testSet = ', list(testSet))
            tempParents = set(parents)
            for parent in parents:
                if parent in testSet:
                    tempParents.remove(parent)
            if not tempParents or nx.d_separated(vg, tempParents, {target}, testSet):
                bSet = list(testSet)
                break
        if verbosity >= 3:
            print('findBackdoorBloockingSet: BDblocking = ', bSet)
        self.bdCache[cacheKey] = bSet
        return bSet

    def findFrontdoorBlockingSet(self, source, target, verbosity=None):
        cacheKey = (source, target)
        if cacheKey in self.fdCache.keys():
            return self.fdCache[cacheKey]
        if verbosity is None:
            verbosity = self.verbosity
        backdoorSet = self.findBackdoorBlockingSet(source, target)
        maxBlocking = 2
        bSet = []
        # Create a graph view that removes the direct link from the source to the destination
        def includeEdge(s, d):
            #print('source, dest = ', s, d)
            if s == source and d == target:
                return False
            return True

        pathNodes = {}
        vg = nx.subgraph_view(self.g, filter_edge=includeEdge)
        # Use that view to find all indirect paths from source to dest
        paths0 = nx.all_simple_paths(vg, source, target)
        paths = [path for path in paths0]
        #print('paths = ', paths)
        if len(paths) == 0:
            # No indirect paths
            return []
        for path in paths:
            #print('path = ', path)
            # Remove the first and last node of the path -- always the source and target
            intermediates = path[1:-1]
            #print('int = ', intermediates)
            for i in intermediates:
                if i not in pathNodes:
                    pathNodes[i] = 1
                else:
                    pathNodes[i] += 1
        pathTups = []
        # First look for single node solutions
        for node in pathNodes.keys():
            cnt = pathNodes[node]
            outTup = (cnt, node)
            pathTups.append(outTup)
               
        # Sort the nodes in descending order of the number of paths containing it
        pathTups.sort()
        pathTups.reverse()
        combos = [(tup[1],) for tup in pathTups]
        #print('pathNodes = ', pathNodes.keys())
        # Now add any multiple field combinations.  Order is not significant here.
        multiCombos = self.getCombinations(pathNodes.keys(), maxBlocking, minOrder=2)
        combos += multiCombos
        #print('combos = ', combos)
        for nodeSet in combos:
            testSet = set(list(nodeSet) + list(backdoorSet))
            #print('testSet = ', list(testSet))
            if nx.d_separated(vg, {source}, {target}, testSet):
                bSet = list(nodeSet)
                break
        if verbosity >= 3:
            print('findFrontdoorBloockingSet: FDblocking = ', bSet)
        self.fdCache[cacheKey] = bSet
        return bSet

