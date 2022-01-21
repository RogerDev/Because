from because.probability import ProbSpace, PDF

class Cprob:
    def __init__(self, dataset, model=None):
        self.ds = dataset
        self.model = model
        self.ps = ProbSpace(self.ds)

    def distr(self, target, cond=None, do=None, cf=None):
        if do is not None or cf is not None:
            assert self.model is not None, "Cprob.distr: A causal model is required in order to use 'do' or 'cf' parameters.  None was specified."
        print('target, cond, do, cf = ', target, cond, do, cf)
        d = self.ps.distr(target)
        return d
