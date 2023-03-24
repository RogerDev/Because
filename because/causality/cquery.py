"""
ProbQuery provides a simplified query interface for ProbSpace.
Queries are submitted as a simple probability string, very similar
to standard mathematical notation for probabilities.

For example:
  'P(age > 50)' # The prob age > 50 (unconditional)
  'P(A=1 | B=0)' # The prob that a is 1 given that b is zero
  'P(size in [large,medium])' # The P that size is 'large' or 'medium'
  'E(income | age between [20,30])' # The expectation of income for 20 to 30 year olds.
  'P(income | age between [20,30])' # The distribution of incomes for 20 to 30 year olds.
  'P(age <= 50, income>5)' # Joint probability
  'E(income | gender=female, age >= 30)' # Expectation with Multiple conditions.
Note that there is no need to quote variable names or values. Strings are automatically
detected by context.

Valid comparison operators are:
  =
  >
  >=
  <
  <=
  between[<low>, <high>]
  in [<val1>, <val2>, ...]

Queries are issued against a ProbSpace instance (ps) containing the targetted dataset.

Queries can be submitted as follows:
  - Single Query -- probquery.query(ps, querystring)
  - Multiple Queries -- proquery.queryList(ps, [querystring1, ...])

query returns either a number (probability or expectation), or a pdf object 
(i.e. a distribution).
Bound target values (e.g. A=1, size in ['small', 'medium']) will result in a
number being returned.  Unbound values (e.g. A, size) will result in a
distribution, similar to standard notation. P(A) is a distribution, while
P(A=1) is a numerical probability.  Expectations (e.g. E(A)) are always
unbound and always return a numeric result.

queryList returns a list containing either numbers or distributions, following
the same rules as above.

Examples:
  from because.probability import probquery

  prob = probquery.query(ps, 'P(A=true | B>2, C<=3)')
  distrs = probquery.queryList(ps, [ 
                'P(income)',
                'P(income | gender=male),
                'P(income | gender=female)'
                ])
  exp = probquery.query(ps, 'E(A | B<-2, C between[-1,1])')
  mixed = probquery.queryList(ps, [
                'P(A)',
                'P(A > 0)',
                'E(A | B=0)'
                ])

The parameter allowedResults (default None) allows the caller to narrow
the set of result types permitted.  This list may contain any subset of
['P', 'E', 'D'], indicating Probability, Expectation, or Distribution
respectively.  Note that type 'D' (distribution) is indicated by a
probability query with an unbound target (e.g. P(income)).  An unbound
target results in a univariate distribution (PDF). 
"""
from because.hpcc_utils.parseQuery import Parser

def queryList(cm, inList, allowedResults=None, power=1, verbosity=0):
    """
    Process a list of (potentially) causal queries and return a list of results.
    """
    ps = cm.prob
    vars = ps.getVarNames()
    specList = PARSER.parse(inList)
    results = []
    for i in range(len(specList)):
        spec = specList[i]
        cmd, targ, cond, ctrlfor, interv, cfac = spec
        assert not cfac, 'Error pasrsing causal query.  Causal query does not yet support' + \
            'counterfactual (i.e. P[<counterfactual>](...) clause).  In:' + inList[i] + '.'
        # We can just append the controlFor variables to the conditional for straight
        # probability queries
        cond += ctrlfor
        for term in targ + cond + interv + cfac:
            var = term[0]
            assert var in vars, 'Error parsing causal query.  Variable name: "' + repr(var) + \
                '" is not in dataset.  In:' + inList[i] + '.  Valid variables are:' + \
                str(vars) + '.'
            vals = term[1:]
            valid = None
            for val in vals:
                if type(val) == type(''):
                    if valid is None:
                        valid = ps.getValues(var)
                    assert val in valid, 'Error parsing probability query. Invalid value:' + \
                    repr(val) + 'for variable:' + var + '. Valid values are:' + str(valid) + \
                    '.'
        if cmd == 'D':
            assert allowedResults is None or cmd in allowedResults,  'Error processing causal query: ' + \
                inList[i] + '.' + '  Target is unbound, resulting in a distribution.  Distribution ' + \
                'type targets are not supported in this context.'
        elif cmd == 'P':
            assert allowedResults is None or cmd in allowedResults,  'Error processing causal query: ' + \
                inList[i] + '.' + '  Target is bound.  Only unbound targets (i.e. distributions) ' + \
                'are allowed in this context.'
        else:
            assert allowedResults is None or cmd in allowedResults,  'Error processing causal query: ' + \
                inList[i] + '.' + '  Result type = ' + cmd + \
                ' is not allowed in this context.  Valid result types are:' + \
                str(allowedResults) + '.'
        if cmd == 'P':
            result = cm.P(targ,cond, interv, power=power, verbosity=verbosity)
        elif cmd == 'E':
            result = cm.E(targ,cond, interv, power=power, verbosity=verbosity)
        elif cmd == 'D':
            result = cm.distr(targ,cond, interv, power=power, verbosity=verbosity)
        results.append(result)
    return results

def query(cm, s, allowedResults=None, power=1, verbosity=0):
    """
    Process a single query and return a single result.
    Parameters and results are the same as for queryList above,
    except that it takes a single query string instead of
    a list, and returns a single result, rather than a list.
    """
    results = queryList(cm, [s], allowedResults, power=power, verbosity=verbosity)
    return results[0]

# Singleton Parser instance
PARSER = Parser()