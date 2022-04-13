def format(funct, targetSpecs, condSpecs=[], doSpecs=[], cfSpecs=[]):
    targetFmt = ''
    condFmt = ''
    doFmt = ''
    cfFmt = ''
    targs = []
    for spec in targetSpecs:
        targName = spec[0]
        if len(spec) == 1:
            targ = targName
        elif len(spec) == 2:
            targ = targName + ' = ' + str(spec[1])
        else:
            targ = str(spec[1]) + ' <= ' + targName + ' < ' + str(spec[2])
        targs.append(targ)
    targetFmt = (', ').join(targs)
    conds = []
    for spec in condSpecs:
        condName = spec[0]
        if len(spec) == 1:
            cond = condName
        elif len(spec) == 2:
            cond = condName + ' = ' + str(spec[1])
        else:
            cond = str(spec[1]) + ' <= ' + condName + ' < ' + str(spec[2])
        conds.append(cond)
    condFmt = (', ').join(conds)
    dos = []
    for spec in doSpecs:
        doName = spec[0]
        assert len(spec) == 2, 'Intervention (Do) specification must be exact match (i.e. have 2 members).'
        do = doName + ' = ' + str(spec[1])
        dos.append(do)
    if dos:
        doFmt = 'do(' + (', ').join(dos) + ')'
    else:
        doFmt = ''
    if doFmt:
        if condFmt:
            condFmt = doFmt + ', ' + condFmt
        else:
            condFmt = doFmt
    cfs = []
    for spec in cfSpecs:
        cfName = spec[0]
        assert len(spec) > 1, 'Counterfactual expression must be bound (i.e. have 2 or 3 members.'
        if len(spec) == 2:
            cf = cfName + ' = ' + str(spec[1])
        else:
            cf = str(spec[1]) + ' <= ' + cfName + ' < ' + str(spec[2])
        cfs.append(cf)
    if cfSpecs:
        cfFmt = '<' + (', ').join(cfs) + '>'
    if condFmt:
        finalFmt = funct + cfFmt +'(' + targetFmt + ' | ' + condFmt + ')'
    else:
        finalFmt = funct + cfFmt + '(' + targetFmt + ')'
    return finalFmt