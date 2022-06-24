from because.probability import prob
from because.synth import gen_data

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
    targetSpec = probspace.normalizeSpecs(targetSpec)
    condSpec = probspace.normalizeSpecs(condSpec)
    dims = len(targetSpec) + len(condSpec)
    assert dims <= 3, 'Vis.show: Can only visualize up to three dimensions.  Got: ' + str(dims) + '.'
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
            pass
        probspace = prob.ProbSpace(ds)
    # Should have probspace at this point.
    valid_gtypes = ['exp', 'pdf', 'cdf', 'joint', 'jointc']
    if gtype == 'exp':
        assert len(targetSpec) == 1, 'Vis.show: Only a single target is supported for Expectation graphs.  Got: ' + str(targetSpec)
        assert len(condSpec) >= 2, 'Vis.show: Expectation graphs must specify 1 or 2 conditions.'
        if dims == 2:
            graph = 'cprobPlot1_5D'
        else:
            graph = 'cprobPlot2_5D'
    else:
        assert len(targetSpec) == 1 or len(condSpec) == 0, 'Vis.show: Currently only support multiple targets with zero conditions,' + \
                            ' or single target with one or two conditions.  Got: '+ \
                            str(len(targetSpec)) + ' targets and ' + str(len(condSpec)) + ' conditions.'