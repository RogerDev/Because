"""
Module for generating multivariate synthetic data.

Data is generated from a python formatted Structural
Equation Model(SEM).  See models/example.py for
details on the model format.
Generates data in .csv format.  Use read_data.py
to read the .csv back into a dataset.
"""
import numpy as np
import math
from math import *
from numpy.random import *
from sys import argv
import random

# Tuning Parameters
MIN_COEF = .1  # Minimum absolute value for coefficients
COEF_SCALE = 1 # Scale of exponential distribution to be added to the MIN_COEF
MEAN_MEAN = 0.0  # Average value of generated noise means
MEAN_SCALE = 5 # Gaussian Standard Deviation to be added to MEAN_MEAN to choose the Mean of the noise
MIN_STD = .1 # Minimum Scale parameter to be applied when generating noise
STD_SCALE = 5 # Scale of exponential distribution to be added to MIN_STD when chosing the scale of the noise.

# DISTRS is a set of distributions to be chosen from for generating noise.  
# # Note distributions must support location and scale parameters, and must allow negative locations			
DISTRS = ['lognormal', 'laplace', 'logistic', 'gumbel', 'normal']
VALIDATION = None
CURR_EQUATION = 0
NOISE_COUNT = 0
NOISES = {}
COEF_COUNT = 0
COEFS = {}
MAX_DIFFICULTY = 0


# Input file name should typically have a .py extension
class Gen:
    """
        Synthetic dataset generator class.
    """
    def __init__(self, semFilePath=None, mod=None, sem=None, init=None):
        """
        Constructor for Gen class.

        Construction can be from a file by supplying semFilePath, or
        from a arguments by passing sem, modm and (optionally) init.

        If from file, then only semFilePath should be provided and other
        arguments should be default (or None).

        For sem file format and format of mod, sem, and init see 
        ./models/example.py.

        Args:
            semFilePath (string): Path to the model file containing the SEM
                    from which to generate the data.
            mod (list): A list of model definitions of the same format as the lines
                in the 'model' parameter of a model file.
            sem (list): A list of equations of the same format as the 'varEquations'
                parameter of a model file
            init (list): A list of initialization lines that are only executed once.
        """
        assert (semFilePath is not None and sem is None and mod is None) or (semFilePath is None and sem is not None and mod is not None), \
                'Gen: Must provide semFilePath or sem and mod, but not both.'
        self.semFileName = semFilePath
        self.sem = sem
        self.mod = mod
        self.varNames = []
        if self.semFileName:
            f = open(self.semFileName, 'r')
            contents = f.read()
            exec(contents, globals())
        else:
            global varEquations, model
            varEquations = self.sem
            model = self.mod
        self.varEquations = varEquations # From tag in the SEM file 'varEquations', a list of equations
        varNames = []
        for rv in model:
            observed = True
            if type(rv) == type((1,)):
                if len(rv) >= 2:
                    name, parents = rv[:2]
                if len(rv) >= 3:
                    observed = rv[2]
                if len(rv) >= 4:
                    datType = rv[3]
            else:
                # Just a list of var names.  No model.
                name = rv
                parents = []
            if observed:
                varNames.append(name)
        self.varNames = varNames
        self.reset = None
        self.init = init

    def generate(self, count=1000, reset=False, quiet=False):
        """
        Function to generate the synthetic dataset and write to an output file
        with the same path as the input file, but with a .csv extension.

        Args:
            count (int, optional): The number of multivariate
                    samples to generate. Defaults to 1000.
            reset (bool, optional): If True, and the SEM uses noise() or
                    coef() terms to generate arbitrary noise or coefficient
                    parameters, will produce a new distribution. If noise() or 
                    coef() are not used, or if set to false, will return new
                    samples from the original distribution. Defaults to False.
            quiet (bool, optional): Suppresses status messages if True.
                    Defaults to False.

        Returns:
            string: The path to the generated .csv data file.
        """
        assert self.semFileName is not None, 'Generate requrires that Gen was constructed with a semFilePath.'
        samples = self.samples(count, reset, quiet)
        # For out file, use the input file name with the .csv extension
        tokens = self.semFileName.split('.')
        outFileRoot = str.join('.',tokens[:-1])
        outFileName = outFileRoot + '.csv'
        outLines = []
        outLine = str.join(',', self.varNames) + '\n'
        outLines.append(outLine)
        for sample in samples:
            sample2 = [str(s) for s in sample]
            outLine = str.join(',', sample2) + '\n'
            outLines.append(outLine)
        f = open(outFileName, 'w')
        f.writelines(outLines)
        f.close()
        return outFileName

    def getDataset(self, count=1000, reset=False, quiet=False):
        """
        Generates synthetic data inline and returns it in dataset format.
        Dataset format is {varName: [samples]}.

        Args:
            count (int, optional): The number of samples to generate. 
                    Defaults to 1000.
            reset (bool, optional): If True, and the SEM uses noise() or
                    coef() terms to generate arbitrary noise or coefficient
                    parameters, will produce a new distribution. If noise() or 
                    coef() are not used, or if set to false, will return new
                    samples from the original distribution. Defaults to False.
            quiet (bool, optional): True to run without status printouts.
                    Defaults to False.

        Returns:
            dictionary: A dictionary of variable name -> list of sample values. 
        """
        dataset = {}
        varNames = self.varNames
        for var in varNames:
            dataset[var] = []
        insamples = self.samples(count, reset, quiet)
        for sample in insamples:
            for i in range(len(sample)):
                val = sample[i]
                dataset[varNames[i]].append(val)
        return dataset

    def samples(self, count=1000, reset=False, quiet=False):
        """
        Generator function to produce the samples.

        Args:
            count (int, optional): The number of multivariate
                    samples to generate. Defaults to 1000.
            reset (bool, optional): If True, and the SEM uses noise() or
                    coef() terms to generate arbitrary noise or coefficient
                    parameters, will produce a new distribution. If noise() or 
                    coef() are not used, or if set to false, will return new
                    samples from the original distribution. Defaults to False.
            quiet (bool, optional): Suppresses status messages if True.
                    Defaults to False.

        Returns:
            list: A list of samples, each containing a list of values,
                    one for each variable.
        """
        global VALIDATION, NOISES, NOISE_COUNT, COEFS, COEF_COUNT, CURR_EQUATION
        global smallestStd, largestStd, smallestCoef, largestCoef, MAX_DIFFICULTY
        locs = locals()
        if self.reset is None:
            # First time.  Treat as if reset is True
            self.reset = True
        else:
            self.reset = reset

        if self.init:
            # run any initialization statements
            for initline in self.init:
                # Don't allow imports
                if 'import' not in initline:
                    try:
                        exec(initline, globals(), locs)
                    except:
                        assert False, 'synth.gen_data:  Bad import line = ' + initline
                else:
                    assert False, 'synth.gen_data: Imports are not allowed in initialization statements. Got = ' + initline

        success = False
        while not success:
            outLines = []
            cEquations = []
            cVarNames = []
            if self.reset:
                NOISES = {}
                COEFS = {}
                smallestStd = 10**100
                largestStd = 0
                smallestCoef = 10**100
                largestCoef = 0
            for eq in self.varEquations:
                if 'import' in eq:
                    assert False, 'synth.gen_data: Imports are not allowed in SEM equations. Got = ' + eq
                else:
                    try:
                        cEquations.append(compile(eq,'err', 'single'))
                    except:
                        assert False, 'synth.gen_data: Bad Equation = ' + eq
            for varName in self.varNames:
                cVarNames.append(compile(varName, 'err', 'eval'))

            for sample in range(count):
                outTokens = []
                NOISE_COUNT = 0
                COEF_COUNT = 0
                for i in range(len(cEquations)):
                    CURR_EQUATION = i
                    try:
                        varEquation = cEquations[i]
                    except:
                        assert False, 'Bad cEquations'
                    try:
                        exec(varEquation, globals(), locs)
                    except:
                        assert False, '*** Invalid Equation = ' + self.varEquations[i]
                for i in range(len(cVarNames)):
                    outTokens.append(eval(cVarNames[i], globals(), locs))
                yield outTokens
            success = True
        return

    def calcOne(self, target, givens=[]):
        """
        Calculate the value of one variable after setting the values of other variables

        Args:
            target (string): A variable name.
            givens (list): A list of tuples (<givenVar>, <givenVal>).  This must include
            any variables upon which the target is dependent.
        Returns:
            float: The value of the target variable.
        """
        locs = locals()
        givenVars = []
        for given in givens:
            var, val = given
            locs[var] = val
            givenVars.append(var)          
        for varEquation in varEquations:
            isGiven = False
            for givenVar in givenVars:
                # Don't re-execute any givens equations.
                if varEquation[:len(givenVar)] == givenVar:
                    isGiven = True
                    break
            if not isGiven:
                exec(varEquation, globals(), locs)
        return eval(target)

    def getVariables(self):
        """
        Return a list of variable names in the same order as the samples.

        Returns:
            list: List of variable names.
        """
        return self.varNames

    def getSEM(self):
        """
        Returns a string representing the SEM used to generate the data.
        This will only return valid results after 'samples' has been called.

        Returns:
            string: Multiline string representing the SEM.
        """
        global NOISE_COUNT, COEF_COUNT
        NOISE_COUNT = 0
        COEF_COUNT = 0
        outEquations = []
        for equation in self.varEquations:
            equation = "'" + equation + "',"
            while equation.find('data()') >= 0:
                equation = equation.replace('data()', str(DATA_OFFSET) + ' + ' + NOISES[NOISE_COUNT],1)
                temp = noise() # to increment the NOISE_COUNT
            while equation.find('coef()') >= 0:
                coefVal = coef()
                equation = equation.replace('coef()',str(coefVal),1)
            while equation.find('noise()') >= 0:
                equation = equation.replace('noise()', NOISES[NOISE_COUNT],1)
                temp = noise() # to increment the NOISE_COUNT
            outEquations.append(equation)
        if largestCoef > 0 or largestStd > 0:
            outEquations.append('Stats:')
            coefRange = largestCoef / float(smallestCoef)
            stdRange = largestStd / float(smallestStd)
            totalRange = coefRange * stdRange
            outEquations.append('  Coef Range = ' + str(coefRange))
            outEquations.append('  Std Range = ' +str(stdRange))
            outEquations.append('  Total Scale Range = ' + str(totalRange))
        outStr = '\n' + str.join('\n', outEquations) + '\n'
        #print(outEquations, outStr)
        return outStr
        

def noise():
    # Generate a random noise distribution and return a sample from it
    global NOISE_COUNT
    currNoise = NOISES.get(NOISE_COUNT, None)
    if currNoise is None:
        distType = random.choice(DISTRS)
        mean = chooseMean()
        std = chooseStd()
        currNoise = distType + '(' + str(mean) + ',' + str(std) + ')'
        NOISES[NOISE_COUNT] = currNoise
        #print('noise[', NOISE_COUNT, '] = ', currNoise)
    NOISE_COUNT += 1
    return eval(currNoise)

DATA_OFFSET = 1.000

def data():
    data = noise() + DATA_OFFSET
    return data
        
# Stats:
smallestStd = 10**100
largestStd = 0
smallestCoef = 10**100
largestCoef = 0

# Remaining functions are used internally
def coef():
    global COEF_COUNT
    global smallestCoef, largestCoef
    coef = COEFS.get(COEF_COUNT, None)
    if coef is None:
        coef = exponential(COEF_SCALE*MIN_COEF) + MIN_COEF
        if coef < smallestCoef:
            smallestCoef = coef
        if coef > largestCoef:
            largestCoef = coef
        sign = random.choice([-1,1,1,1,1])
        coef *= sign
        coef = round(coef, 3)
        COEFS[COEF_COUNT] = coef
    COEF_COUNT += 1
    return coef



def chooseMean():
    return round(normal(MEAN_MEAN, MEAN_SCALE),3)


def chooseStd():
    global smallestStd, largestStd
    val = round(exponential(STD_SCALE * MIN_STD) + MIN_STD,3)
    if val < smallestStd:
        smallestStd = val
    if val > largestStd:
        largestStd = val
    return val

# If run from command line, parameters are filepath and datacount
if __name__ == '__main__':
    filename = None
    if len(argv) <= 1 or '-h' in argv:
        print('Usage: python gen_data.py filepath [datasize].')
        print('     If datasize is not specified, defaults to 1000.')
    else:
        filename = argv[1]
        if len(argv) > 2:
            datacount = eval(argv[2])
        else:
            datacount = 1000
        print('Generating data for model = ', filename)
        #print('filename, datacount = ', filename, datacount)
        if filename is not None:
            gen = Gen(filename)
            gen.generate(count=datacount)
        print ('SEM = ', gen.getSEM())
        print('Generated ', datacount, 'records.')

	