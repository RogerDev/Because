from because.hpcc_utils import formatQuery

tests = [('P', [('A', 1)]),
        ('P', [('A', 1), ('B',2)]),
        ('P', [('A', 1), ('B',2)], [('C', 3), ('D', 1,2)]),
        ('E', [('A',)], [('C', 3), ('D', 1,2)]),
        ('E', [('A',)], [('B',),('C', 3), ('D', 1,2)]),
        ('Distr',[('A',)], [('B',),('C', 3), ('D', 1,2)]),
        ('Distr',[('A',)], [], [('B', .5),('C', 3), ('D', 1)]),
        ('Distr',[('A',)], [('C', 3), ('D', 1,2)], [('B', 1), ('F', -1)]),
        ('Distr',[('A',)], [('C', 3), ('D', 1,2)], [], [('C', 0), ('D', 0)]),
        ]

format = formatQuery.format

for test in tests:
    assert len(test) > 1 and len(test) <= 5, 'Bad Test Definition = ' + str(test)
    if len(test) == 2:
        result = format(test[0], test[1])
    elif len(test) == 3:
        result = format(test[0], test[1], test[2])
    elif len(test) == 4:
        result = format(test[0], test[1], test[2], test[3])
    else:
        result = format(test[0], test[1], test[2], test[3], test[4])

    print(result)