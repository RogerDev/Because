"""
Produce a directed acyclic graph of a set of variables' causal relationships.
"""
from sys import argv, path
if '.' not in path:
    path.append('.')
import time
from math import log, tanh, sqrt, sin, cos, e

import matplotlib.pyplot as plt
import networkx as nx

from because.synth import read_data, gen_data
from because.probability.prob import ProbSpace
from because.causality import cdisc
from because.utils import vprint

def show(dataPath='', numRecs=0, targetSpec=[], condSpec=[], controlFor=[], gtype='pdf', probspace=None, cg=None, edgeLabels=None, enhance=True, 
        power=5, maxLevel=2, sensitivity=5, verbosity=3):
    assert (cg is not None and probspace is None and not dataPath) or \
            (cg is None and probspace is not None and not dataPath) or \
            (cg is None and probspace is None and dataPath and numRecs > 0), \
            'cmodel.show: Must provide a cGraph, or ProbSpace instance, or a dataPath and numRecs.' 
    #fig = plt.figure()
    if edgeLabels == 'None':
        # Default edge labels to correlation
        edgeLabels = 'corr'
    fig, axs = plt.subplots(1,1)
    ax = axs
    ax.set_axis_off()
    ax.set_frame_on(False)
    start = time.time()
    tx = None
    if cg is None and probspace is None:
        # Got a .csv or .py file
        tokens = dataPath.split('.')
        ds = None # The dataset in dictionary form
        assert len(tokens) == 2 and (tokens[1] == 'py' or tokens[1] == 'csv'), 'Cmodel.show: dataPath must have a .py or .csv extension.  Got: ' + dataPath
        vprint(3, verbosity, 'cmodel.show: Processing data file = ', dataPath)
        if tokens[1] == 'py':
            # py SEM file
            assert numRecs > 0, 'Cmodel.show: For synthetic data (i.e. .py extension) numRecs must be positive'
            gen = gen_data.Gen(dataPath)
            ds = gen.getDataset(numRecs)
        else:
            # csv
            r = read_data.Reader(dataPath)
            ds = r.read()
        prob1 = ProbSpace(ds, power=power)
    elif cg is None:
        # Use ProbSpace
        prob1 = probspace
    else:
        # Use cGraph
        prob1 = cg.prob
        targetSpec = cg.rvList # Ignore targetSpec when cg provided.
    if not targetSpec:
        targets = prob1.getVarNames()
    else:
        targetSpec = prob1.normalizeSpecs(targetSpec)
        targets = [spec[0] for spec in targetSpec]
    
    if cg is None:
        vprint(3, verbosity, 'cmodel.show: Performing Discovery.')
        # If we didn't get a cgraph, discover one.
        cg = cdisc.discover(prob1, varNames=targets, maxLevel=maxLevel, power=power, sensitivity=sensitivity, verbosity=verbosity)
    
    edges = []
    edgeLabelDict = {}
    edgeVals = {}

    vprint(3, verbosity, 'cmodel.show: Analyzing Graph Relations.')
    rvs = cg.getRVs()
    if edgeLabels == 'mde':
        elText = 'Maximum Direct Effect'
    elif edgeLabels == 'rho':
        elText = 'Rho-value'
    else:
        elText = 'Correlation'
    vprint(4, verbosity, 'Showing', elText, 'for edge labels')
    for rv in rvs:
        var = rv.name
        parents = rv.parentNames
        for parent in parents:
            link = (parent, var)
            corr = prob1.corrCoef(parent, var)
            if edgeLabels == 'mde':
                effect = cg.MDE(parent, var)
                if corr < 0:
                    effect *= -1
            elif edgeLabels == 'rho':
                effect = cg.getEdgeProp(link, 'dir_rho')
            else:
                effect = corr
            edgeVals[link] = effect
            edgeLabelDict[link] = str(round(effect, 2))
            edges.append(link)
            # Discovery puts the directional rho as a edge property.  Retrieve it.
            dir = cg.getEdgeProp(link, 'dir_rho')
            if dir is None:
                vprint(2, verbosity, 'cmodel.show: Missing edge property "dir_rho"')
                continue
            if abs(dir) < .1:
                # Very weak directional signal.  Add a reverse link to graph.
                vprint(4, verbosity, 'cmodel.show: Adding bidirectional link for', link)
                v1, v2 = link
                edges.append((v2, v1))
                edgeLabelDict[(v2, v1)] = ''
                edgeVals[(v2, v1)] = 0
    # Build networkx graph
    vprint(3, verbosity, 'cmodel.show: Producing graphics.')
    gr = nx.DiGraph()
    gr.add_nodes_from(targets)
    gr.add_edges_from(edges)
    cmapName = 'BrBG'
    cmap = plt.get_cmap(cmapName)
    nodeCmap = plt.get_cmap('tab20')
    nodeColors = nodeCmap.colors
    nodeColors = [nodeColors[i % len(nodeColors)] for i in range(len(targets))]

    def normColors(val):
        if val > 0:
            color = cmap(.6 + val * .4)
        elif val < 0:
            color = cmap(.4 + val * .4)
        else:
            color = [.5,.5,.5,.5]
        return tuple(list(color))

    edgeColors = []
    for edge in gr.edges():
        u, v = edge
        edgeColor = normColors(edgeVals[(u,v)])
        edgeColors.append(edgeColor)
    ax.clear()
    ax.set_axis_off()
    ax.set_frame_on(False)
    #ax = fig.get_axes()[0]
    pos = nx.circular_layout(gr)
    #pos = nx.kamada_kawai_layout(gr)
    #pos = nx.spectral_layout(gr)
    #pos = nx.planar_layout(gr)
    #pos = nx.spring_layout(gr)
    if verbosity >= 1:
        print('Map = ',)
        for edge in gr.edges():
            v1, v2 = edge
            strength = edgeVals[(v1, v2)]
            if edgeLabelDict[edge] != '': # Eliminate reverse links
                print('  ', v1, '-->', v2, '(', round(strength, 3), ')')
    nx.draw(gr, pos=pos, ax=ax, with_labels=True, node_size=1500, arrows=True, arrowsize=30, node_color=nodeColors, edge_color=edgeColors, width=2.0)
    nx.draw_networkx_edge_labels(gr, pos, ax=ax, edge_labels=edgeLabelDict)
    plt.title("Causal Relationships", fontsize=16)
    end = time.time()
    vprint(2, verbosity, 'cmodel.show: Elapsed = ', round(end - start, 0))
    plt.show()

if __name__ == '__main__':
    if '-h' in argv or len(argv) < 5:
        print('Usage: python because/visualization/cmodel.py dataPath targets power pruneLevel [numRecs]')
        print('  dataPath is the path to a .py (synthetic data) or .csv file')
        print('  targets is the variable(s) whose distribution to plot.')
        print('  sensitivity is the sensitivity to use for ProbSpace dependence methods.')
        print('  maxLevel is the number of conditionals to consider when pruning the graph [0,3]')
        print('  numRecs is the number of records to generate (for synthetic data only)')
    else:
        numRecs = 0 
        args = argv
        dataPath = args[1].strip()
        targets = args[2].strip()
        sensitivity = int(args[3].strip())
        tokens = targets.split(',')
        tSpec = []
        for token in tokens:
            varName = token.strip()
            if varName:
                tSpec.append((varName,))
        if len(args) > 4:
            maxLevel = int(args[4].strip())
        if len(args) > 5:
            try:
                numRecs = int(args[5].strip())
            except:
                pass
       
        #print('dims, datSize, numRecs = ', dims, datSize, numRecs)
        show(dataPath=dataPath, numRecs=numRecs, targetSpec=tSpec, power=5, maxLevel=maxLevel, sensitivity=sensitivity)



