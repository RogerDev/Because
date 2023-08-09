import pandas as pd
import numpy as np
from because.probability.ucm import ucm
from because.synth import Reader
'''
UCM testing on CDC dataset. 
Tests all pairs of directions present in the categorical variables. 
I recommend saving output to a text file on run for analysis: 
    python3 llcp_pairs.py > output.txt
Computing with full dataset vs a sample will produce somewhat different results. 
'''

df = pd.read_csv("models/llcp.csv")

ds = Reader("models/llcp.csv")
# ds = Reader("models/llcp.csv", limit=10000)

varNames = ds.getSeriesNames()
numeric = ['age', 'weight', 'height', 'bmi']
discreteNum = ['ageGroup', 'income', 'sleephours', 'drinks']
catVars = [i for i in varNames if i not in (numeric+discreteNum)]

for i in range(len(catVars)):
    # print(f'\n\tvariable: {catVars[i]}, length: {len(set(ds.getSeries(catVars[i])))} \n {set(ds.getSeries(catVars[i]))}')
    for j in range(i + 1, len(catVars)):
        A = np.array(df[catVars[i]].tolist())
        B = np.array(df[catVars[j]].tolist())
        print(f"\n=============\n\ttesting: {catVars[i]} -> {catVars[j]}")
        rho, identifiable = ucm.uniform_channel_test(A, B)
        if identifiable:
            if rho > 0:
                print(f"Implied Direction: {catVars[i]} -> {catVars[j]}")
            else:
                print(f"Implied Direction: {catVars[j]} -> {catVars[i]}")
