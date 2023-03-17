"""
ProbQuery provides a simplified query interface for ProbSpace and Causality.
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
  'E(income | do(gender=female,veteran=yes), age>=30)' # Causal Intervention using do() semantic
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

Note that for do() interventions, only equality (=) is supported.


"""
class Parser:
    """
    Class to parse the query syntax 
    """
    def parse(self, strList, isGraph=False):
        """
        Main entry point.  Parse a list of query strings and return a list of:
        (queryType, targetSpec, conditionSpec, controlForSpec, doSpec, counterfacSpec).

        The most general form of a query is:
        query := <queryType>[<counterfactual>](<targetclause> | <condclause>)
        condclause := <clause>, controlFor(<clause>), do(<clause>) 
        clause := <variableName> <oper> <value>, ...
        oper := 1 of ['=', '>', '<', '>=', '<=', 'between', 'in']
        value := <simpleVal> or <valList>
        simpleVal := number or stringValue
        valList := [<simpleVal>, ...]

        Example:
            E(A | B between [0.0, 3.0])
            # Expected value of A given that B is between 0 and 3.
            P[C=1, D=2](A in [1,2,3] | C=0, D=1, do(E=3), controlFor(F,G))
            # Probability (in a world where C=1 and D=2) that A=1, 2 , or 3,
            # given that C=0 and D=1, when we intervene to make E = 3, and
            # controlFor F and G.

        queryTypes are:
        - P -- Probability
        - E -- Expectation
        - D -- Distribution

        Specs are formatted as per ProbSpace queries.
        """
        if type(strList) == type(''):
            strList = [strList]
        outList = []
        for s in strList:
            doSpec = [] # do(...)
            ctrlSpec = [] # controlFor(...)
            cfacSpec = [] # counterfactual clause
            orig = s
            pos = s.find('(')
            assert pos >= 0, 'Error parsing probability query.  No opening parentheses in:' + orig
            startclause = s[:pos].strip()
            cfacstart = startclause.find('[')
            if cfacstart >= 0:
                # We have a counterfactual
                # Strip it out, and process the counterfactual clause to a spec.
                cfacend = startclause.find(']', cfacstart+1)
                assert cfacend >= 0, 'Error parsing probability query.  No closing bracket on counterfactual in: ' + orig 
                cfacclause = startclause[cfacstart+1:cfacend]
                outtype = startclause[:cfacstart].strip()
                cfacterms = self.parseTerms(cfacclause)
                cfacSpec = self.convertToSpec(cfacterms)
            else:
                outtype = startclause.strip()
            s = s[pos+1:]
            pos = s.rfind(')')
            assert pos >=0, 'Error parsing probability query.  No closing parentheses in:' + orig
            s = s[:pos]
            tok = s.split('|')
            targ = tok[0].strip()
            if len(tok) > 1:
                cond = tok[1].strip()
            else:
                cond = ''
            targ = self.fixupBrak(targ)
            # Separate the do() and controlFor() clauses (i.e. interventions and controls)
            # from the rest of the conditional clause.  Example:  P(A | do(B=1,C=2), controlFor(E), D=3)
            cond, interv, ctrlfor = self.extractSubspecs(cond)
            cond = self.fixupBrak(cond)
            tterms = self.parseTerms(targ)
            tSpec = self.convertToSpec(tterms)
            if interv:
                dterms = self.parseTerms(interv)
                doSpec = self.convertToSpec(dterms)
            if ctrlfor:
                ctrlterms = self.parseTerms(ctrlfor)
                ctrlSpec = self.convertToSpec(ctrlterms)
            cterms = self.parseTerms(cond)
            cSpec = self.convertToSpec(cterms)
            outtype = outtype.upper()
            assert outtype in ['P', 'E', 'DEPENDENCE', 'CORRELATION', 'CMODEL'], \
                'Error parsing probability query.  Invalid result type:"' + \
                outtype + '". Valid types are "P" for probability or "E" for expectation, ' + \
                    'Dependence, Correlation, or Cmodel'
            if outtype == 'P' and (len(tSpec) == 1 or (len(tSpec) == 2 and isGraph and len(tSpec[1]) == 1)) and len(tSpec[0]) == 1:
                # It is probability with an unbound target. The answer is
                # a distribution
                outtype = 'D'
            outList.append((outtype, tSpec, cSpec, ctrlSpec, doSpec, cfacSpec))
            #print('outList = ', outList)
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

    def extractSubspecs(self, s):
        """
        Separate the do() clause from the condition clause (if do() present) and;
        Separate the controlFor() clause (if controlFor() is present)
        """
        doclause = ''
        ctrlclause = ''
        condclause = s
        dostart = s.find('do(')
        if dostart >= 0:
            doend = s.find(')', dostart)
            assert doend >= 0, 'Error parsing probability query.  Unterminated do clause in:' + repr(s) + '.'
            condclause = (s[0:dostart] + s[doend+1:]).strip()
            doclause = s[dostart+3:doend]
        ctrlstart = condclause.find('controlFor(')
        if ctrlstart >= 0:
            ctrlend = condclause.find(')', ctrlstart)
            assert ctrlend >= 0, 'Error parsing probability query.  Unterminated controlFor clause in:' + repr(s) + '.'
            ctrlclause = condclause[ctrlstart+11 : ctrlend]
            condclause = (condclause[0:ctrlstart] + condclause[ctrlend+1:]).strip()
        return condclause, doclause, ctrlclause

    def fixupType(self, item):
        try:
            outItem = float(item)
        except ValueError:
            outItem = item
        return outItem

    def fixupTypes(self, l):
        outl = []
        for item in l:
            item = self.fixupType(item)
            outl.append(item)
        return outl

    def parseTerms(self, s):
        # cases = '',' in ', ' between ', '<', '>', '='
        splitCases = ['>=','>', '<=', '=', '<', ' in ', ' between ']
        terms = []
        if s:
            # Convert => to >= and =< to <=
            s = s.replace('=>', '>=')
            s = s.replace('=<', '<=')
            toks = s.split(',')
            for tok in toks:
                tcase = ''
                tok = tok.strip()
                if not tok:
                    # Ignore tokens with no content.
                    continue
                var = tok
                val = None
                for c in splitCases:
                    t2 = tok.split(c)
                    if len(t2) > 1:
                        tcase = c
                        var = t2[0].strip()
                        val = t2[1].strip()
                        if tcase == ' in ' or tcase == ' between ':
                            # break up the list
                            val = val[1:-1].split(';')
                            val = [v.strip() for v in val]
                            val = self.fixupTypes(val)
                        else:
                            val = self.fixupType(val)
                        break
                        
                terms.append((tcase, var, val))
        return terms

    def convertToSpec(self, inTerms):
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
            elif ttype == ' between ':
                outTerm = (term[1], term[2][0], term[2][1])
            elif ttype == ' in ':
                #outTerm = (term[1],) + tuple(term[2])
                outTerm = (term[1], tuple(term[2]))

            else:
                outTerm = ('ERROR', term[1], term[2])
            outTerms.append(outTerm)
        return outTerms

