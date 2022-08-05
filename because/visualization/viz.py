import time
from sys import argv
from because.probability import prob
from because.synth import gen_data, read_data
from because.visualization import probPlot1D, probPlot2D_exp, probPlot2D, probPlot3D_exp, probPlot3D, probPlot2D_bound, probPlot3D_bound, probPlotAll

def show(dataPath='', numRecs=0, targetSpec=[], condSpec=[], filtSpec=[], controlFor=[], gtype='pdf', probspace=None, enhance=False, power=None):
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
    assert probspace is not None or dataPath, 'Viz.show:  Must specify either dataPath or probspace.'
    valid_gtypes = ['exp', 'pdf', 'cdf', 'multi']
    assert gtype in valid_gtypes, 'Vizshow:  Invalid gtype provided.  Valid types are: ' + str(valid_gtypes) + '. Got: ' + str(gtype)
    tempSpec = []
    for targ in targetSpec:
        if type(targ) == type(''):
            tempSpec.append((targ,))
        else:
            tempSpec.append(targ)
    targetSpec = tempSpec

    targetsAreBound = max([len(targ) for targ in targetSpec]) > 1
    assert not targetsAreBound or gtype != 'exp', 'Vis.show: Bound targets are incompatible with expectation (gtype=exp) graphs.'
    assert not targetsAreBound or gtype != 'multi', 'Vis.show: Bound targets are incompatible with multi-variable (gtype=multi) graphs.'
    if probspace is None:
        # Got a .csv or .py file
        tokens = dataPath.split('.')
        ds = None # The dataset in dictionary form
        assert len(tokens) == 2 and (tokens[1] == 'py' or tokens[1] == 'csv'), 'Viz.show: dataPath must have a .py or .csv extension.  Got: ' + dataPath
        if tokens[1] == 'py':
            # py SEM file
            assert numRecs > 0, 'Viz.show: For synthetic data (i.e. .py extension) numRecs must be positive'
            gen = gen_data.Gen(dataPath)
            ds = gen.getDataset(numRecs)
        else:
            # csv
            r = read_data.Reader(dataPath)
            ds = r.read()
        probspace = prob.ProbSpace(ds)
    
    # Should have probspace at this point.
    # First filter it.
    filtSpec = probspace.normalizeSpecs(filtSpec)
    if len(filtSpec) > 0:
        print('Viz.show: Filtering by: ', filtSpec)
        probspace = probspace.SubSpace(filtSpec)
        print('Viz.show: Filtered N = ', probspace.N)
    targetSpec = probspace.normalizeSpecs(targetSpec)
    condSpec = probspace.normalizeSpecs(condSpec)
    print('Viz.show: Target = ', targetSpec, ', Condition = ', condSpec)
    dims = len(targetSpec) + len(condSpec)
    assert dims <= 3 or gtype == 'multi', 'Viz.show: Can only visualize up to three dimensions.  Got: ' + str(dims) + '.'
    valid_gtypes = ['exp', 'cum']
    if gtype == 'multi':
        assert len(condSpec) == 0, 'Viz.show: Conditions cannot be used for multi-variable graphs (gtype=multi).'
        graph = probPlotAll
        graphName = 'Multi Variable PDF Plot'
    elif gtype == 'exp':
        assert len(targetSpec) == 1, 'Viz.show: Only a single target is supported for Expectation graphs.  Got: ' + str(targetSpec)
        assert len(condSpec) > 0 and len(condSpec) <= 2, 'Vis.show: Expectation graphs must specify 1 or 2 conditions.  Got: ' + str(len(condSpec))
        if dims == 2:
            graph = probPlot2D_exp
            graphName = '2D Expected value plot with single condition'
        else:
            graph = probPlot3D_exp
            graphName = '3D Expected value plot with two conditions'
    elif targetsAreBound:
        assert len(condSpec) == 1 or len(condSpec) == 2, 'Viz.show: Bound Targets must specify one or two conditionals.'
        if len(condSpec) == 1:
            graph = probPlot2D_bound
            graphName = '2D Bound Probability plot with 1 conditional.'
        else:
            graph = probPlot3D_bound
            graphName = '3D Bound Probability plot with 2 conditionals.'
    else:
        assert len(targetSpec) == 1 or len(condSpec) == 0, 'Viz.show: Currently only support multiple targets with zero conditions,' + \
                            ' or single target with one or two conditions.  Got: '+ \
                            str(len(targetSpec)) + ' targets and ' + str(len(condSpec)) + ' conditions.'
        if dims == 1:
            graph = probPlot1D
            graphName = 'Univariate Probability Distribution plot.'
        elif dims == 2:
            graph = probPlot2D
            graphName = '2-variable Probability Distribution plot.'
        else:
            graph = probPlot3D
            graphName = '3-variable Probability Distribution plot.'
    print('Viz.show: Showing graph = ', graphName, ', power = ', power)
    if graph != probPlot1D and graph != probPlotAll:
        if controlFor:
            print('Viz.show: Controlling for ', controlFor)
        graph.show(targetSpec=targetSpec, condSpec=condSpec, controlFor=controlFor, gtype=gtype, probspace=probspace, enhance=enhance, power=power)
    else:
        graph.show(targetSpec=targetSpec, condSpec=condSpec, gtype=gtype, probspace=probspace)

if __name__ == '__main__':
    if '-h' in argv or len(argv) < 4:
        print('Usage: python because/visualization/viz.py dataPath targets conditions filters controlFor [numRecs] [cum] [enh]')
        print('  dataPath is the path to a .py (synthetic data) or .csv file')
        print('  targets is the variable(s) whose distribution to plot e.g., A, "A,B".')
        print('  conditions are the conditional variable name(s) e.g., B, "B,C".')
        print('  numRecs is the number of records to generate')
        print('  filters is a list of (varName, value) to filter the dataset by before processing')
        print('  controlFor is a list of variable names to conditionalize on.')
        print('  "cum" as a final parameter causes display of cumulative distributions.  Otherwise PDFs are shown.')
        print('  "enh" as a final parameter causes enhanced probability graphs to be shown.')
    else:
        numRecs = 0 
        args = argv
        dataPath = args[1].strip()
        targets = args[2].strip()
        conditions = args[3].strip()
        filters = args[4].strip()
        controlFor = args[5].strip()
        if targets[0] == '(' or targets[0] == '[':
            # Targets is specified as a tuple or list. Probably bound.
            # Just eval to interpret it.
            tSpec = eval(targets)
            if targets[0] == '(':
                tSpec = [tSpec]
        else:
            # Targets is specified as a var name, or comma separated set of var names
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
        if filters:
            filtSpec = eval(filters)
        else:
            filtSpec = []
        cfSpec = []
        if controlFor:
            tokens = controlFor.split(',')
            for token in tokens:
                varName = token.strip()
                if varName:
                    cfSpec.append(varName)
        if len(args) > 6:
            try:
                numRecs = int(args[5].strip())
            except:
                pass
        gtype = 'pdf'
        if 'cum' in args:
            gtype = 'cdf'
        if 'exp' in args:
            gtype = 'exp'
        enhance = False
        if 'enh' in args:
            enhance = True
        if 'multi' in args:
            gtype = 'multi'
        
        #print('dims, datSize, numRecs = ', dims, datSize, numRecs)
        show(dataPath=dataPath, numRecs=numRecs, targetSpec=tSpec, 
            condSpec=cSpec, filtSpec=filtSpec, controlFor=cfSpec, gtype=gtype, enhance=enhance, power=5)
