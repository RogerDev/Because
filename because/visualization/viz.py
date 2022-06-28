from sys import argv
from because.probability import prob
from because.synth import gen_data, read_data
from because.visualization import probPlot1D, probPlot2D_exp, probPlot2D, probPlot3D_exp, probPlot3D

def show(dataPath='', numRecs=0, targetSpec=[], condSpec=[], gtype='pdf', probspace=None):
    """
    Must specify either:
    - dataPath -- A .py SEM file for synthetic data or .csv for other data
    - probspace -- An instanted prob.ProbSpace object
    numRecs is only for generating synthetic data
    targetSpec and condSpec are the same as for ProbSpace
    Graph type (gtype) values are:
    - 'pdf' -- Probability Density Function
    - 'cdf' -- Cumulative Probability Density
    - 'exp' -- Expected Value
    """
    assert probspace is not None or dataPath, 'Vis.show:  Must specify either dataPath or probspace.'
    valid_gtypes = ['exp', 'pdf', 'cdf']
    assert gtype in valid_gtypes, 'Vis.show:  Invalid gtype provided.  Valid types are: ' + str(valid_gtypes) + '. Got: ' + str(gtype)
    if probspace is None:
        # Got a .csv or .py file
        tokens = dataPath.split('.')
        ds = None # The dataset in dictionary form
        assert len(tokens) == 2 and (tokens[1] == 'py' or tokens[1] == 'csv'), 'Vis.show: dataPath must have a .py or .csv extension.  Got: ' + dataPath
        if tokens[1] == 'py':
            # py SEM file
            assert numRecs > 0, 'Vis.show: For synthetic data (i.e. .py extension) numRecs must be positive'
            gen = gen_data.Gen(dataPath)
            ds = gen.getDataset(numRecs)
        else:
            # csv
            r = read_data.Reader(dataPath)
            ds = r.read()
        probspace = prob.ProbSpace(ds)
    # Should have probspace at this point.
    targetSpec = probspace.normalizeSpecs(targetSpec)
    condSpec = probspace.normalizeSpecs(condSpec)
    dims = len(targetSpec) + len(condSpec)
    assert dims <= 3, 'Vis.show: Can only visualize up to three dimensions.  Got: ' + str(dims) + '.'
    valid_gtypes = ['exp', 'cum']
    if gtype == 'exp':
        assert len(targetSpec) == 1, 'Vis.show: Only a single target is supported for Expectation graphs.  Got: ' + str(targetSpec)
        assert len(condSpec) > 0 and len(condSpec) <= 2, 'Vis.show: Expectation graphs must specify 1 or 2 conditions.  Got: ' + str(len(condSpec))
        if dims == 2:
            graph = probPlot2D_exp
        else:
            graph = probPlot3D_exp
    else:
        assert len(targetSpec) == 1 or len(condSpec) == 0, 'Vis.show: Currently only support multiple targets with zero conditions,' + \
                            ' or single target with one or two conditions.  Got: '+ \
                            str(len(targetSpec)) + ' targets and ' + str(len(condSpec)) + ' conditions.'
        if dims == 1:
            graph = probPlot1D
        elif dims == 2:
            graph = probPlot2D
        else:
            graph = probPlot3D
    
    graph.show(targetSpec=targetSpec, condSpec=condSpec, gtype=gtype, probspace=probspace)

if __name__ == '__main__':
    if '-h' in argv or len(argv) < 4:
        print('Usage: python because/visualization/viz.py dataPath targets conditions [numRecs] [cum]')
        print('  dataPath is the path to a .py (synthetic data) or .csv file')
        print('  targets is the variable(s) whose distribution to plot e.g., A, "A,B".')
        print('  conditions are the conditional variable name(s) e.g., B, "B,C".')
        print('  numRecs is the number of records to generate')
        print('  "cum" as a final parameter causes display of cumulative distributions.  Otherwise PDFs are shown')
    else:
        numRecs = 0 
        args = argv
        dataPath = args[1].strip()
        targets = args[2].strip()
        conditions = args[3].strip()
        tokens = targets.split(',')
        tSpec = []
        for token in tokens:
            varName = token.strip()
            if varName:
                tSpec.append((varName,))
        tokens = conditions.split(',')
        cSpec = []
        for token in tokens:
            varName = token.strip()
            if varName:
                cSpec.append((varName,))
        if len(args) > 4:
            try:
                numRecs = int(args[4].strip())
            except:
                pass
        gtype = 'pdf'
        if 'cum' in args:
            gtype = 'cdf'
        if 'exp' in args:
            gtype = 'exp'
        
        #print('dims, datSize, numRecs = ', dims, datSize, numRecs)
        show(dataPath=dataPath, numRecs=numRecs, targetSpec=tSpec, condSpec=cSpec, gtype=gtype)
