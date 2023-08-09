import pprint
from because.probability.prob import ProbSpace
from because.synth import Reader
from because.visualization import cmodel
from because.probability.ucm.ucm import UCVar, CategVar

# Example of using the UCVar Class

'''
-- Initialize the variables with a base distribution and any parent variables. 
Base distributions should be a real probability distribution, with values adding to 1. 
Parent variables should be initialized first, followed by child variables. 
The parents of each variable should be passed when initializing each child variable. 

'''

varA = UCVar(base_distribution=[0.8, 0.2], values=['male', 'female'])
varB = UCVar(base_distribution=[0.1, 0.1, 0.8], values=['small', 'medium', 'large'], parent=varA)
varC = UCVar(base_distribution=[0.4, 0.6], values=['yes', 'no'], parent=[varA, varB])

'''
For each variable, we are able to print the corresponding output values, distribution, or parent values. 
'''
print(f"\nParent Values (possible inputs): \n\t{varC.parent_vals}")
print(f"\nPossible output values: \n\t{varC.values}")
print("\nDistribution of C (given (A,B) pair) : \n\t")
pprint.pprint(varC.distribution)

'''
Generating values. 
If the variable is endogenous, then pass the values of the parent variables 
(and in the same order as they were passed during initialization). 
'''

A = varA.get_value()
B = varB.get_value(A)
C = varC.get_value([A, B])

print(f"\nGenerated values: \n\tA:{A},  B:{B}, C:{C}")

'''
Output values which are not initialized will be inferred from the base distribution. 
'''
varD = UCVar(base_distribution=[0.25, 0.25, 0.5],  parent=varB)
print(f"\nPossible output values: {varD.values}")

'''
There is also a CategVar class, with distribution explicitly defined by parent vars.
Intened to be a non-UC categorical variable but wasn't used much; not well-fleshed out.
'''
dist = {
    'male': [0.8, 0.2],
    'female': [0.6, 0.4]
}
varE = CategVar(distribution=dist, parent=varA)

print("\nDistribution of E (given A) : \n\t")
pprint.pprint(varE.distribution)
print(f"\nPossible output values: {varE.values}")

''' 
For doing testing and discovery, read dataset into a ProbSpace object and use testDirection. 
'''
# Read in model, define models variables
filename = "probability/ucm/models/M2.csv"
ds = Reader(filename)
ps = ProbSpace(ds.varData, categorical=['A', 'B', 'C'])

# To test a particular direction
out = ps.testDirection('A', 'B')
print(f"\nCalculated rho value: {out}")

# To perform discovery on target variables and show discovered the causal graph
cmodel.show(probspace=ps, targetSpec=['A','B','C'], sensitivity=10, power=5, edgeLabels='rho', verbosity=1)
