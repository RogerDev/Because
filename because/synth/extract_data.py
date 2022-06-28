
def extract(infile, outfile, fields):
    recsOut = 0
    f = open(infile, 'r')
    firstLine = f.readline()
    tokens = firstLine.split(',')
    fieldIndexes = {}
    indexFields = {}
    for i in range(len(tokens)):
        token = tokens[i].strip()
        if token in fields:
            fieldIndexes[token] = i
            indexFields[i] = token
    f2 = open(outfile, 'w+')
    header = ','.join(fields) + '\n'
    f2.write(header)
    while True:
        fieldDict = {}
        line = f.readline()
        tokens = line.split(',')
        for i in range(len(tokens)):
            if i in indexFields.keys():
                token = tokens[i]
                try:
                    val = float(token)
                except:
                    assert True, 'extract_data: error parsing line number ' + i + '.  Contents = ' + token
                fieldDict[indexFields[i]] = token
        if len(fieldDict.keys()) != len(fields):
            if len(line) == 0:
                break
            print('skipping line = ', line)
            continue
        outFields = []
        for field in fields:
            outFields.append(fieldDict[field])
        outLine = ','.join(outFields)
        f2.write(outLine + '\n')
        recsOut += 1
    print('Wrote', recsOut, 'records.')
    f.close()
    f2.close()

if __name__ == '__main__':
    from sys import argv
    inFile = argv[1]
    outFile = argv[2]
    fieldsRaw = argv[3]
    fields = []
    tokens = fieldsRaw.split(',')
    for token in tokens:
        fields.append(token.strip())
    extract(inFile, outFile, fields)