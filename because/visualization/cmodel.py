"""
Produce a directed acyclic graph of a set of variables' causal relationships.
"""
from sys import argv, path
if '.' not in path:
    path.append('.')
import time
from math import log, tanh, sqrt, sin, cos, e

import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as anim
import networkx as nx

from because.causality import rv
from because.synth import read_data, gen_data
from because.probability import independence
from because.probability.prob import ProbSpace
from because.probability.rkhs.rkhsMV import RKHS
from because.visualization import grid2 as grid
from because.causality import rv
from because.causality import calc_indeps

def show(dataPath='', numRecs=0, targetSpec=[], condSpec=[], controlFor=[], gtype='pdf', probspace=None, enhance=True, power=1, prunelevel=2, verbosity=2):
    #fig = plt.figure()
    fig, axs = plt.subplots(1,1)
    ax = axs
    ax.set_axis_off()
    ax.set_frame_on(False)
    start = time.time()
    tx = None
    if probspace is None:
        # Got a .csv or .py file
        tokens = dataPath.split('.')
        ds = None # The dataset in dictionary form
        assert len(tokens) == 2 and (tokens[1] == 'py' or tokens[1] == 'csv'), 'Heatmap.show: dataPath must have a .py or .csv extension.  Got: ' + dataPath
        if tokens[1] == 'py':
            # py SEM file
            assert numRecs > 0, 'Heatmap.show: For synthetic data (i.e. .py extension) numRecs must be positive'
            gen = gen_data.Gen(dataPath)
            ds = gen.getDataset(numRecs)
        else:
            # csv
            r = read_data.Reader(dataPath)
            ds = r.read()
        prob1 = ProbSpace(ds, power=power)
    else:
        prob1 = probspace
    targetSpec = prob1.normalizeSpecs(targetSpec)
    targets = [spec[0] for spec in targetSpec]
    pairs = []
    for i in range(len(targets)):
        for j in range(i+1, len(targets)):
            pair = (targets[i], targets[j])
            pairs.append(pair)        
    for level in range(prunelevel+1):
        if verbosity >= 1:
            print('Executing Pass', level)
        parentDict = {}
        for target in targets:
            parentDict[target] = []
        edgeValsDict = {}
        for pair in pairs:
            if tx is not None:
                tx.set(visible = True)
                plt.pause(.001)
            indeps = calc_indeps.calculateOne(prob1, pair[0], pair[1], targets, power=power, maxLevel=level)
            if tx is not None:
                tx.set(visible = False)
                plt.pause(.1)
                tx.set(visible = True)
                plt.pause(.001)
            valid = True
            for indep in indeps:
                item, isInd = indep
                if isInd:
                    # There's at least one case that renders this link invalid.
                    if verbosity >= 1:
                        print('   link ', pair[0], '---', pair[1], 'pruned by: ', item[2])
                    valid = False
                    break
            if not valid:
                continue
            # Not independent or conditionally independent.  Direct causal link.  Test direction.
            dir = prob1.testDirection(pair[0], pair[1], power=power, N_train=10000)
            corr = prob1.corrCoef(pair[0], pair[1])
            if dir > 0:
                if verbosity >= 2:
                    print('   link ', pair[0], '-->', pair[1], 'is causally valid.')
                parentDict[pair[1]].append(pair[0])
            elif dir < 0:
                if verbosity >= 2:
                    print('   link ', pair[1], '-->', pair[0], 'is causally valid.')
                parentDict[pair[0]].append(pair[1])
            else:
                continue
            edgeValsDict[(pair[0], pair[1])] = corr
            edgeValsDict[(pair[1], pair[0])] = corr
    
        edges = []
        edgeLabels = {}

        for child in parentDict.keys():
            parents = parentDict[child]
            for parent in parents:
                edges.append((parent, child))
                edgeVal = edgeValsDict[(parent, child)]
                edgeLabels[(parent, child)] = str(round(edgeVal, 2))
        end = time.time()
        print('Elapsed = ', round(end - start, 0))
        # Build networkx graph
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
                color = cmap(.75 + val/4.0)
            elif val < 0:
                color = cmap(.25 + val/4.0)
            else:
                color = [0,0,0,0]
            return tuple(list(color))

        edgeColors = []
        for edge in gr.edges():
            u, v = edge
            edgeColor = normColors(edgeValsDict[(u,v)])
            edgeColors.append(edgeColor)
        ax.clear()
        ax.set_axis_off()
        ax.set_frame_on(False)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        tx = ax.text(.05, .95, "Pass " + str(level+1) + ' of ' + str(prunelevel) , transform=ax.transAxes, fontsize=14,  color='red', bbox=props)
        #ax = fig.get_axes()[0]
        pos = nx.circular_layout(gr)
        #pos = nx.kamada_kawai_layout(gr)
        #pos = nx.spectral_layout(gr)
        #pos = nx.planar_layout(gr)
        #pos = nx.spring_layout(gr)

        if level == prunelevel:
            if verbosity >= 1:
                print('Map = ',)
                for edge in gr.edges():
                    v1, v2 = edge
                    print('  ', v1, '-->', v2, '(', round(edgeValsDict[(v1, v2)], 3), ')')
            tx.set(visible = False)
        else:
            tx.set(visible = True)
        nx.draw(gr, pos=pos, ax=ax, with_labels=True, node_size=1500, arrows=True, arrowsize=30, node_color=nodeColors, edge_color=edgeColors, width=2.0)
        nx.draw_networkx_edge_labels(gr, pos, ax=ax, edge_labels=edgeLabels)
        plt.title("Causal Relationships", fontsize=16)
        #a = anim.FuncAnimation(fig, update, frames=10, repeat=False)
        plt.draw()
        plt.pause(.001)
    plt.show()

if __name__ == '__main__':
    if '-h' in argv or len(argv) < 5:
        print('Usage: python because/visualization/cmodel.py dataPath targets power pruneLevel [numRecs]')
        print('  dataPath is the path to a .py (synthetic data) or .csv file')
        print('  targets is the variable(s) whose distribution to plot.')
        print('  power is the power to use for ProbSpace methods.')
        print('  pruneLeve is the number of conditionals to consider when pruning the graph [0,3]')
        print('  numRecs is the number of records to generate (for synthetic data only)')
    else:
        numRecs = 0 
        args = argv
        dataPath = args[1].strip()
        targets = args[2].strip()
        power = int(args[3].strip())
        tokens = targets.split(',')
        tSpec = []
        for token in tokens:
            varName = token.strip()
            if varName:
                tSpec.append((varName,))
        if len(args) > 4:
            pruneLevel = int(args[4].strip())
        if len(args) > 5:
            try:
                numRecs = int(args[5].strip())
            except:
                pass
       
        #print('dims, datSize, numRecs = ', dims, datSize, numRecs)
        show(dataPath=dataPath, numRecs=numRecs, targetSpec=tSpec, power=power, prunelevel=pruneLevel)



