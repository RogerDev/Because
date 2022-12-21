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
"""
def queryList(ps, inList):
    """
    Process a list of queries and return a list of results.
    """
    vars = ps.getVarNames()
    specs = PARSER.parse(inList)
    results = []
    for i in range(len(specs)):
        spec = specs[i]
        cmd, targ, cond = spec
        for term in targ + cond:
            var = term[0]
            assert var in vars, 'Error parsing probability query.  Variable name: "' + repr(var) + \
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
        if cmd == 'P':
            result = ps.P(targ,cond)
        elif cmd == 'E':
            result = ps.E(targ,cond)
        elif cmd == 'D':
            result = ps.distr(targ,cond)
        results.append(result)
    return results

def query(ps, s):
    """
    Process a single query and return a single result.
    """
    results = queryList(ps, [s])
    return results[0]

class Parser_:
    """
    Class to parse the query syntax 
    """
    def fixupBrak(self, s):
        orig = s
        pos1 = 0
        pos2 = 0
        while pos1 >= 0:
            pos1 = s.find('[', pos2)
            if pos1 >= 0:
                pos2 = s.find(']', pos1)
                assert pos2 >= 0, 'Error parsing probability query.  Mismatched brackets in:' + orig
                if pos2 >= 0:
                    inner = s[pos1+1:pos2]
                    inner = inner.replace(',',';')
                    s = s[:pos1 + 1] + inner + s[pos2:]                
        return s

    def fixupType(self, item):
        if item.isnumeric():
            item = float(item)
        return item

    def fixupTypes(self, l):
        outl = []
        for item in l:
            item = fixupType(item)
            outl.append(item)
        return outl

    def parseTerms(self, s):
        # cases = '','in', 'between', '<', '>', '='
        splitCases = ['>=','>', '<=', '=', '<', 'in', 'between']
        terms = []
        if s:
            # Convert => to >= and =< to <=
            s = s.replace('=>', '>=')
            s = s.replace('=<', '<=')
            toks = s.split(',')
            for tok in toks:
                tcase = ''
                tok = tok.strip()
                var = tok
                val = None
                for c in splitCases:
                    t2 = tok.split(c)
                    if len(t2) > 1:
                        tcase = c
                        var = t2[0].strip()
                        val = t2[1].strip()
                        if tcase == 'in' or tcase == 'between':
                            # break up the list
                            val = val[1:-1].split(';')
                            val = [v.strip() for v in val]
                            val = self.fixupTypes(val)
                        else:
                            val = self.fixupType(val)
                        break
                        
                terms.append((tcase, var, val))
        return terms

    def convertFinalTerms(self, inTerms):
        inf = 999999999
        outTerms = []
        for term in inTerms:
            ttype = term[0]
            if ttype == '':
                outTerm = (term[1],)
            elif ttype == '>':
                outTerm = (term[1], term[2]+1/inf, inf)
            elif ttype == '>=':
                outTerm = (term[1], term[2], inf)
            elif ttype == '<':
                outTerm = (term[1], -inf, term[2])
            elif ttype == '<=':
                outTerm = (term[1], -inf, term[2]+1/inf)
            elif ttype == '=':
                outTerm = (term[1], term[2])
            elif ttype == 'between':
                outTerm = (term[1], term[2][0], term[2][1])
            elif ttype == 'in':
                outTerm = (term[1],) + tuple(term[2])
            else:
                outTerm == ('ERROR', term[1], term[2])
            outTerms.append(outTerm)
        return outTerms

    def parse(self, strList):
        if type(strList) == type(''):
            strList = [strList]
        outList = []
        for s in strList:
            orig = s
            pos = s.find('(')
            assert pos >= 0, 'Error parsing probability query.  No opening parentheses in:' + orig
            outtype = s[:pos].strip()
            s = s[pos+1:]
            pos = s.find(')')
            assert pos >=0, 'Error parsing probability query.  No closing parentheses in:' + orig
            s = s[:pos]
            tok = s.split('|')
            targ = tok[0].strip()
            if len(tok) > 1:
                cond = tok[1].strip()
            else:
                cond = ''
            targ = self.fixupBrak(targ)
            cond = self.fixupBrak(cond)
            tterms1 = self.parseTerms(targ)
            tterms = self.convertFinalTerms(tterms1)

            cterms1 = self.parseTerms(cond)
            cterms = self.convertFinalTerms(cterms1)
            assert outtype in ['P', 'E'], \
                'Error parsing probability query.  Invalid result type:"' + \
                outtype + '". Valid types are "P" for probability or "E" for expectation.'
            if outtype == 'P' and len(tterms) == 1 and len(tterms[0]) == 1:
                # It is probability with an unbound target. The answer is
                # a distribution
                outtype = 'D'
            outList.append((outtype, tterms, cterms))
        return outList

    def fixupBrak(self, s):
        orig = s
        pos1 = 0
        pos2 = 0
        while pos1 >= 0:
            pos1 = s.find('[', pos2)
            if pos1 >= 0:
                pos2 = s.find(']', pos1)
                assert pos2 >= 0, 'Error parsing probability query.  Mismatched brackets in:' + orig
                if pos2 >= 0:
                    inner = s[pos1+1:pos2]
                    inner = inner.replace(',',';')
                    s = s[:pos1 + 1] + inner + s[pos2:]                
        return s

    def fixupType(self, item):
        if item.isnumeric():
            item = float(item)
        return item

    def fixupTypes(self, l):
        outl = []
        for item in l:
            item = self.fixupType(item)
            outl.append(item)
        return outl

    def parseTerms(self, s):
        # cases = '','in', 'between', '<', '>', '='
        splitCases = ['>=','>', '<=', '=', '<', 'in', 'between']
        terms = []
        if s:
            # Convert => to >= and =< to <=
            s = s.replace('=>', '>=')
            s = s.replace('=<', '<=')
            toks = s.split(',')
            for tok in toks:
                tcase = ''
                tok = tok.strip()
                var = tok
                val = None
                for c in splitCases:
                    t2 = tok.split(c)
                    if len(t2) > 1:
                        tcase = c
                        var = t2[0].strip()
                        val = t2[1].strip()
                        if tcase == 'in' or tcase == 'between':
                            # break up the list
                            val = val[1:-1].split(';')
                            val = [v.strip() for v in val]
                            val = self.fixupTypes(val)
                        else:
                            val = self.fixupType(val)
                        break
                        
                terms.append((tcase, var, val))
        return terms

    def convertFinalTerms(self, inTerms):
        inf = 999999999
        outTerms = []
        for term in inTerms:
            ttype = term[0]
            if ttype == '':
                outTerm = (term[1],)
            elif ttype == '>':
                outTerm = (term[1], term[2]+1/inf, inf)
            elif ttype == '>=':
                outTerm = (term[1], term[2], inf)
            elif ttype == '<':
                outTerm = (term[1], -inf, term[2])
            elif ttype == '<=':
                outTerm = (term[1], -inf, term[2]+1/inf)
            elif ttype == '=':
                outTerm = (term[1], term[2])
            elif ttype == 'between':
                outTerm = (term[1], term[2][0], term[2][1])
            elif ttype == 'in':
                outTerm = (term[1],) + tuple(term[2])
            else:
                outTerm == ('ERROR', term[1], term[2])
            outTerms.append(outTerm)
        return outTerms

    def parse(self, strList):
        if type(strList) == type(''):
            strList = [strList]
        outList = []
        for s in strList:
            orig = s
            pos = s.find('(')
            assert pos >= 0, 'Error parsing probability query.  No opening parentheses in:' + orig
            outtype = s[:pos].strip()
            s = s[pos+1:]
            pos = s.find(')')
            assert pos >=0, 'Error parsing probability query.  No closing parentheses in:' + orig
            s = s[:pos]
            tok = s.split('|')
            targ = tok[0].strip()
            if len(tok) > 1:
                cond = tok[1].strip()
            else:
                cond = ''
            targ = self.fixupBrak(targ)
            cond = self.fixupBrak(cond)
            tterms1 = self.parseTerms(targ)
            tterms = self.convertFinalTerms(tterms1)

            cterms1 = self.parseTerms(cond)
            cterms = self.convertFinalTerms(cterms1)
            assert outtype in ['P', 'E'], \
                'Error parsing probability query.  Invalid result type:"' + \
                outtype + '". Valid types are "P" for probability or "E" for expectation.'
            if outtype == 'P' and len(tterms) == 1 and len(tterms[0]) == 1:
                # It is probability with an unbound target. The answer is
                # a distribution
                outtype = 'D'
            outList.append((outtype, tterms, cterms))
        return outList

# Singleton Parser instance
PARSER = Parser_()