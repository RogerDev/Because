# Causality Framework

The HPCC Systems Causality Framework "_Because_" provides a comprehensive toolkit for causal analysis of multivariate datasets, and for the testing of causal algorithms.  Causal analysis of a dataset is necessary in order to extract meaning from the realationships among the data elements.  Causal analysis requires a broad set of statistical tools, as well as additional tools from the causal realm. The difference between statistical and causal tools is that causal tools interact with a Causal Model, which is represented as a Directed Acyclic Graph (DAG).  Causal tools may attempt to infer a DAG from the data, or may utilize a DAG alongside the data to draw conclusions that cannot be made from the dataset alone.  The former approach is known as Causal Discovery.  The later is known as Causal Inference.
For details on Causality, Causal Inference, and the various causality algorithms, please see the References section below.

## Installation
### Prerequisites
- python 3.6 or greater
- pip3

### Procedure
- Clone repository
- cd repo_path
- sudo -H pip3 install .

This will create the package 'because', which can then be imported.

## Sub-Packages
The Because framework consists of four main sub-packages that can be used independently or in concert.
- _Synthetic Dataset Generator_ -- (from because import synth) provides a flexible synthetic multivariate data generation system.  It can be used, for example, to generate datasets with a known causal mechanism for validating causal algorithms.
- _Probability_ -- (from because import probability) a powerful probability engine that supports a variety of advanced statistical capabilities.
- _Visualization_ -- (from because import visualization) a high-level charting system providing many different views of a dataset, both statistical and causal.
- _Causality_ -- (from because import causality) contains various causal methods including Causal Discovery and Causal Inference.

Each of these submodules is defined below.

## Synthetic Data Generation (synth)

The "models" folder contains various test models including a set of standard test models M0 - M10.  These are synthetic data models, each of which contains a Structural Equation Model (SEM) that is used to generate a dataset with known causality, as well as a Causal Model that represents an hypothesis as to the data generation process.  Generally these two aspects are set up to be consistent, but can be made inconsistent to e.g., see the result of the model validation process.  The SEM can be linear or non-linear, and can include any mathematical functions, or noise distributions (e.g. Normal, Logistic, Exponential, Beta, LogNormal).  The Causal Model may contain variables not described by the SEM, in which case, those variables are treated as "unobserved".  The command line utility synth/synthDataGen.py (see below) is used to process the SEM and generate datasets of any size.  These are produced as CSV datasets.  The synth/getData.py module is used to read these CSV datasets.  Synth can be used to generate a wide variety of datasets including:
- Independent and Identically Distributed (i.i.d.) Datasets
- Time Series Datasets
- Heteroskedastic Datasets

Synthetic data can be generated from a "Synth File" or from an inline structural equation model.

The output can be a .csv file, a set of samples, or an inline Dataset compatible with the other "because" modules.

### Inline data generation

    from because.synth import Gen

    # The sem provides a step by step recipe to generate a sample 
    # for each variable as a list of strings.  In this case, A is
    # sampled from a standard Normal distribution (mean 0, sigma 1).
    # B is from a Logistic distribution (mean 1, scale 2).  C is
    # a function of A and B plus some Normal noise.

    sem = [ 'A = normal(0, 1)', 
            'B = logistic(1, 2)', 
            'C = tanh(A + B) + normal(0, .5)'
          ] 

    # The model contains three variables, A, B and C.  Note that
    # the SEM may contain variables that are not exposed in the
    # model (i.e. intermediate, latent, or hidden variables).
    model = ['A', 'B', 'C']
    
    # Create a generator instance given the model and the sem
    gen = Gen(mod=model, sem=sem)

    # Generate a dataset of 100 records.  Datasets are returned as a dictionary of variable name -> list of observations.
    ds = gen.getDataset(100)

    # Generate a .csv file with 1000 records
    fileName = gen.generate(1000. outFile=<myfilepath>)

    # Generate a list of 10000 sample tuples where each sample
    # is a tuple in the same order as the variables are listed in 
    # model.
    tuples = gen.samples(10000)

### Generating data from a "synth file"

As an alternative to in-line model generation, models can be generated using a pre-constructed "synth file".  These files contain both a SEM and a model component. See synth/models/example.py for an annotated example of a model file. See because/models for a set of various synth files.

    from because.synth import Gen
    
    # Create a Gen instance using the synth filename.
    gen = Gen(semFilePath=<path to my synth file>)

Outputs can be generated using any of forms as above.  If a .csv file is desired, generate will default to the synth file path with a .csv extension.

    fileName = gen.generate(100000)

### Generating synthetic data from the command line

    python synth/gen_data.py <model file> <numRecords>

Example:

    python synth/gen_data.py models/M3.py 100000
Genrates 100,000 multivariate records as described by M3.py

## Probability Methods

_probability/prob.py_ provides a powerful Multivariate Probability Space object (ProbSpace).  It is used by the various causal methods (above), but it can be used as a stand-alone probability layer as well.  It equally supports both discrete (including binary and categorical) and continous data, or any mixture of the above.

Input to ProbSpace is a Dataset in the form of a dictionary: {varName:samples, ...}.  This dataset can be read from a .csv file contaning sythetic or natural data using the synth.read_data() module, or constructed by other means.


    from because.synth import read_data
    from because.probability import ProbSpace
    r = read_data.Reader('<mydatapath>.csv')
    ds = r.read()
    ps = ProbSpace(ds)

ProbSpace allows a wide range of queries including:
### Numerical Probabilities, Joint Probabilities, and Conditional Probabilities, including Conditionalization
- P(X=x) -- The numeric probability that X takes on the value x.
- P(X=x, Y=y) -- Joint Probability -- The probability that X=x and Y=y simultaneously.
- P(Y=y | X=x) -- The numerical probability that Y takes on the value y, given that X is observed to have the value x.
- P(Y=y | X=x, ... , Z=z) -- The numerical probability that Y takes on the value y, given the values of multiple variables.
- P(Y=y | X) -- The numerical probability that Y=y when conditionalized on X.  This is equivalent to the sum of P(Y=y | X=x) * P(X=x), for every value x of X.
- P(Y=y | X=x, Z) -- The numerical probability that Y=y,  given that X=x and conditionalized on Z. There may be additional givens, and multiple variables may be conditionalized on.

Examples:

    p = ps.P(('A', 5)) # Probability that A = 5.
    p = ps.P(('A', -1, 1)) # Probability that -1 <= A < 1.
    p = ps.P(('B', None, 0)) # Probability that B is < 0.
    p = ps.P(('B', 0, None)) # Probability that B is >= 0
    p = ps.P(('A',5), ('B', None, 0)) # P that A = 5 given that B < 0.
    # P that A is between 0 and 1 given that B is between -.5 and .5.
    p = ps.P(('A', 0, 1), ('B', -.5, .5))
    # The probability that A is 1, 3, or 5.
    p = ps.P(('A', [1, 3, 5]))
    # Joint probability that both A and B are non-negative,
    p = ps.P([('A', 0, None), ('B', 0, None)])
    # Joint probability that both A and B are non-negative given
    # that C and D are both zero.
    p = ps.P([('A', 0, None), ('B', 0, None)], [('C', 0), ('D', 0)])
    # The expected value of A given B = 2
    m = ps.E('A', ('B', 2))
    # The expected value of A given B = 2, and C = 3, controlled for D
    m = ps.E('A', [('B', 2), ('C', 3), 'D'])

Note for P(), either the target and the condition can be specified as either a tuple or a list of tuples, and that each tuple can contain the variable and a value, or the variable and a list of valid values (discrete only) or a variable and a minimum value and a maximum value.
For expectation, the target should be unbound (i.e. a bare variable).

### Probability Distributions and Conditional Probability Distributions
The distr() function returns a univariate distribution (pdf) object based on the query.  This can be a marginal or conditional distribution from the dataset.  The resulting distribution can then be queried for a wide range of statistical information (see PDF Objects below).

- P(X) -- The probability distribution of any variable X.  Presented as a PDF (Probability Distribution Function), which is actually a discretized version of the probability distribution.
- P(Y | X=x) -- Conditional Probability -- The probability distribution of Y given that X is observed to have the value x.
- P(Y | X=x, ... , Z=z) -- The probability distribution of Y, given the values of multiple variables.
- P(Y | X) -- The probability distribution of Y when conditionalized on X.  This is equivalent to the sum of P(Y | X=x) * P(X=x), for every value x of X.
- P(Y | X=x, ... , Z=z) -- The conditional probability distribution of Y, given the values of multiple variables.
- P(Y | X=x, Z) -- The probability distribution of Y given that X=x and conditionalized on Z.  There may be additional givens, and multiple variables may be conditionalized on.

Examples:

    # Distribution of A
    d = ps.distr('A')
    # Distribution of A given B >= 0.
    d = ps.distr('A', ('B', 0, None))
    # Distribution of A given B >= 0 and C is 2 or 3.
    d = ps.distr('A', [('B', 0, None), ('C', [2,3])])
    # Distribution of A given B < 0, controlled for
    # (i.e. conditionalized on) C.
    d = ps.distr('A', [('B', None, 0), 'C'])

### PDF Objects
_Probability/pdf.py_ Provides a Probability Distribution Function object, which is a finely discretized probability distribution.  This is returned by Prob (above) whenever a probability distribution is requested.  The following functions are available for a PDF:
- pdf.P(v) -- The probability of a value "v" occurring.
- pdf.E() -- The Expected Value of the distribution (i.e. mean).  This is the first "standard moment" of the distribution.
- pdf.var() -- The Variance of the distribution.
- pdf.stDev() -- The Standard Deviation (sigma) of the distribution. This is the second standard moment of the distribution.
- pdf.skew() -- The skew (third standard moment) of the distribution.
- pdf.kurtosis() -- The excess kurtosis (fourth standard moment) of the distribution.
- pdf.median() -- The median of the distribution.
- pdf.mode() -- The mode of a discrete distibution.
- pdf.percentile(p) -- The p-th percentile of the distribution
- pdf.compare(otherPdf) -- Compare two PDFs, and assess their level of similarity.

Examples:

    mu = d.E() # Expected value (mean) of the distribution
    p = d.P(3) # Probability that the variable will take the value 3.
    p = d.P((-1, 1)) # Probability of -1 <= variable < 1.
    var = d.var() # Variance
    std = d.stDev() # Standard Deviation
    sk = d.skew() # Skew
    ku = d.kurtosis() # Kurtosis
    m = d.median() # Median
    v = d.percentile(99) # 99th percentile
    m = d.mode() # Mode
    # Determine if distribution is bounded (upper or lower), and
    # the position of the bound if bounded.
    upper, lower = d.truncation()
    # Modality of the distribution (e.g. unimodal, bimodal, etc.)
    modes = d.modality()
    # Histogram of the distribution.
    # (returns [(binStart, binEnd, probability), ...])
    hist = d.ToHistTuple()

### Conditional SubSpaces

Conditional SubSpaces are multivariate subspaces of the original distribution, after conditioning on one or more variables.  They contain the same set of variables as the original distribution, but only the samples that meet the conditional filter.  The SubSpace is actually a new ProbSpace object, and therefore has the full capabilities, including further subspacing.

Examples:

    # The subspace where C is negative.
    ps2 = ps.SubSpace(('C', None, 0))
    # The subspace where C is negative and D is 'yes' or 'no'.
    ps2 = ps.Subspace([('C', None, 0), ('D', ['yes', 'no'])])

SubSpace can be used e.g., to clean data by providing filters, or to assess the impact of a conditioning operation on multiple variables.

### Dependency Testing and Conditional Dependency Testing

Rapid Dependence and Independence testing is provided using hierarchical stochastic sampling of dense regions searching for dependence.  A "power" parameter determines the extent of sampling done.  Low power is sufficent for most continuous dependencies, where higher power settings may be needed to detect complex discontinuous dependence.  Higher powers take exponentially longer to complete.

- Dependence(X, Y) -- The numerical dependency measured between X and Y.
- Dependence(X, Y | Z=z, W=w) -- The numerical dependency measured between X and Y given one or more observed covariate values.
- Dependence(X, Y | Z=z, W) -- The numerical dependency measured between X and Y, given that Z=z and conditionalized on W.  There may be multiple conditional values or variables to conditionalize on.

### Independence Testing and Conditional Independence Testing

- Independence(X, Y | Z=z, W) -- All the flavors of conditional independence, as with dependence (above), but inverted.
- isIndependent(X, Y | Z=z, W) -- All flavors of conditional independence. returns Boolean, True if independent, otherwise False.

### Prediction and Classification

This is similar to a Machine Learning system, except that there is no training.  All predictions are done at query time rather than using a training process.  It can handle any type of dependent and independent variables (binary, categorical, continuous) and can handle any type of relationships among data elements (linear, non-linear, continuous, discontinuous), making it more flexible than most ML methods.  It is also very easy to use because there is no need to encode or standardize data, and no assumptions are made regarding the relationships or disturbances.  Generalization is handled via progressive filtering.

- PredictDist(Y | X1=x1, X2=x2, ... , XN=xN) -- Returns the distribution (see PDF below) of a dependent variable (Y) given any number of independent variables (X) and their values. 
- Predict(Y | X1=x1, X2=x2, ... , XN=xN) -- Perfoms a non-linear regression operation, providing a best estimate of the dependent variable (Y) given any number of
    independent variables and their values.
- Classify(Y | X1=x1, X2=x2, ... , XN=xN) -- Choses the most likely categorical class (Y=y)  given any number of independent variables and their values.

### Command Line Programs

_probTest.py_ is a comprehensive test for the probability system, and demonstrates the use of the probability system using synthetic data.  It is hardcoded to use the data file models/probTestDat.py.

    python probability/test/probTest.py

_probPredTest.py_ exercises the Prediction capabilities of the Probability module for both Regression and Classification.

    python probability/test/probPredTest.py

_indCalibrate.py_ is used to calibrate the thresholds for independence testing.

    python  probabilty/test/indCalibrate.py

## Visualization

The visualization sub-system (_because.visualization_) provides a library of visualization methods geared toward understanding the variables in your dataset and their relationships.

The module _viz.py_ provides a common interface into most of the visualization methods.  Many types of visualization are supported:

- Univariate probability charts showing attributes of a variable's distribution along with its Probability Density Function (PDF) or its Cumulative Distribution Function (CDF).
- Comparative charts showing the distributions of multiple variables.
- Heatmaps showing the Correlation or Dependence among many variables simultaneously.
- Two-variable joint distribution chart. 3-D visualization showing the joint distribution of a pair of variables (joint PDF or CDF).
- Conditional Expectations.  The expected value of one variable (target) given the value of a conditional variable.
- Conditional Probability.  3-D chart showing the probability distribution of a target variable given the value of a conditional variable.
- Two-variable conditional expectation.  3-D chart showing the expected value of a target variable for each value of two conditional variables.
- Controlled Conditional Probabilities -- The probability distribution of a target variable conditional on the value of another variable, while controlling for one or more additional variables.
- Controlled Conditional Expectations -- The expected value of a target variable for each value of one or two conditional variables, while controlling for one or more additional variables.
- Causal Model Chart -- A graphical representiation of a discovered or constructed causal model.  Includes analysis of the strength and direction of causal influence.

Examples:

    # Assume we already have a constructed ProbSpace called "ps"
    # as in Probability Methods above.

    from visualization import viz

    # Show a PDF chart of variable X1
    viz.show(probspace=ps, targetSpec=['X1'], gtype='pdf')
    
    # Show the PDFs of multiple variables.
    viz.show(probspace=ps, targetSpec=['X1', 'X2', 'X3', 'X4'],     gtype='multi')

    # Show the conditional expectation of general health given income.
    viz.show(probspace=ps, targetSpec=['genhealth'], condSpec=['income'], gtype='exp')

    # Same as the above, but filtered only for males.
    viz.show(probspace=ps, targetSpec=['genhealth'], condSpec=['income'], filtSpec=[('gender', 'male')], gtype='exp')

    # Same as above, but controlled for age and state of residence.
    viz.show(probspace=ps, targetSpec=['genhealth'], condSpec=['income'], filtSpec=[('gender', 'male')], controlFor=['age', 'state'], gtype='exp')

    # Show a heatmap of multiple variables
    viz.show(probspace=ps, targetSpec=['age', 'gender', 'height', 'weight', 'bmi', 'veteran', 'smokertype', 'physicalactivity', 'income', 'education', 'genhealth'], gtype='hmap')

### Viz Arguments

As shown above, viz provides a flexible set of arguments that determine the type of graph shown.

- targetSpec -- a list of one or more target variable names or names and value specifications (for bound targets).  These take the same form as target specifications in ProbSpace:
    - [varName] -- Variable is the target.
    - [(varName, value)] -- Variable = value is the target.
    - [(varName, startVal, endVal)] -- startVal <= Variable < endVal is the target. 
    - [(varName, [val1, val2, ... ])] -- variable in [val1, val2, ...] is the target.
- condSpec -- a list of one or two variable names upon which to condition the target variable.
- filtSpec -- a list of filters to apply to the dataset before charting the remainder of records.  Allowed forms are:
    - [(varName, value)] -- Variable == value
    - [(varName, startVal, endVal)] -- startVal <= Variable < endVal. 
    - [(varName, [val1, val2, ... ])] -- variable in [val1, val2, ...].
- controlFor -- A list of variable names to control for (i.e. conditionalize on).  This is only useful for conditional charts, where it shows (target | condition) while controlling for control variables.
- gtype -- The type of graph to generate.  Allowed values are:
    - 'pdf' -- PDF of the target.
    - 'cdf' -- CDF of the target.
    - 'exp' -- Expected value of the target.
    - 'hmap' -- Heatmap chart
    - 'multi' -- Multiple PDF chart.
- power -- A value between 0.0 and 10.0 that is passed directly to ProbSpace for all statistical calculations.  It allows a tradeoff between runtime and depth of analysis for calculations.  For example, a power of zero utilizes linear methods while a power of 10.0 utilizes the most sophisticated non-linear methods, but can require much higher runtime.  For most cases a power of 3 to 5 is sufficient.  One approach is to do preliminary analysis with a low power, and move to a higher power for final precise values.  One special use of power is that a heatmap with power=0 will show intervariable correlation (positive and negative), while powers >= 1 will show non-linear dependence, which is always non-negative.
- enhance -- True or False.  Default False. Affects only 3-D charts.  True enhances the readability of the chart by eliminating noisy low probability areas.

### Causal Model Visualizations

Causal Model Visualizations are not supported by viz.  They use a separate module called "cmodel".

Causal Models can be automatically discovered from a ProbSpace instance, or can be used to show a previously created or discoverd cGraph instance (see cGraph in Causality Methods below).

Examples:

    from because.visualization import cmodel

    # Causal model of four variables discovered from a ProbSpace instance 
    cmodel.show(probspace=ps, targetSpec=['gender', 'height', 'weight', 'genhealth'], sensitivity=5, power=5, edgeLabels='mde', verbosity=1)

    # Causal model from an already existing cGraph.
    cmodel.show(cg=mycgraph, edgeLabels='corr')

### Causal Model Visualizaton Arguments

The following arguments control cmodel.show()

- probspace -- The probspace object from which to discovery, if provided.  Not provided when using a prepopulated cGraph.
- targetSpec -- A list of variable names to discover.  Not used when using a prepopulated cGraph.
- cg -- a prepopulated cGraph instance.
- sensitivity -- A value from 1.0 to 10.0 that is passed directly to causal discovery (see cDisc under Causal Methods below).  Not used for prepopulated cGraphs.
- maxLevel -- Tha maximum number of conditions to evaluate for dependency testing.  Default=2.  Not used for prepopulated cGraphs.
- power -- Power level (0.0 to 10.0) as defined for ProbSpace above.
- edgeLabels -- The meaning of the labels on graph edges.  Allowed values are "corr" for correlation, or "mde" for maximum direct effect.
- verbosity -- a verbosity level for diagnostic information, from 0 to 4.  Zero is silent while 4 provides detailed diagnostic information.

### Running Visualizations from the Command Line

A subset of visualization capabilities are available to run from the command line.

    python3 <path-to-because>/visualization/viz.py dataPath targets conditions filters controlFor [numRecs] [cum] [exp] [multi] [enh]

For example:

    cd Because/because
    // 3D chart of A given B and C filtered by D > 0, based on a
    // synthetic data generation of 10000 records.
    python3 visualization/viz.py mySynthFile.py A "B,C" "[(D, 0, None)]" "" 10000 exp

    // Multip PDF chart of age, weight, and height, using data from
    // a .csv file.
    python3 visualization/viz.py myDatFile.csv "age,weight,height" "" "" "" multi


## Causal Methods

_RV_ provides a class representing a Random Variable.  A Causal Model is composed of two or more inter-related random variables.

_cGraph.py_ (Causal Graph) is the heart of the Causal Subsystem.  It accepts a multivariate dataset as well as a Causal Model (aka Path Diagram) that is thought to correspond to the causal process that produced the data.  The Causal Model is a Directed Acyclic Graph that represents the best hypothesis of the causal process that underlies the generation of the provided dataset.  The Causal Model may contain "unobserved" variables as well as the "observed" variables with their corresponding datasets.  cGraph includes the following functional areas:
- Model Editing -- The ability to edit a previously provided model, by adding or removing variables or relationships, reversing causal directions, etc.
- Model Testing -- Analyzes the data and highlights areas where the actual data deviates from the Causal Model's Hypotheses.  It enumerates the independence implications of the Causal Model, and verifies each of these expected independencies / dependencies using various conditional independence testing methods.  It also uses LiNGAM algorithms to test the directionalities of each link in the model.  It combines the results of these different tests to provide a composite numerical assessment of the match between the Causal Model and the Data.
- Intervention -- Implements the "Do Calculus", for causal inference, emulating the results of a controlled experiment.  The Do Calculus attempts to screen-off non-causal correlation and identify the effect of an intervention (i.e. setting a variable to a fixed value) on other variables. If the model does not provide sufficient observed variables to screen-off the non-causal effects, than the result is returned as "not identifiable".
- Causal Metrics -- Various Causal Metrics can be requested regarding a pair of variables including: Average Causal Effect (of X on Y), Controlled Direct Effect, Maximum Causal Effect, Maximum Direct Effect, and Indirect Effect.
- Counterfactual Inference (Future) -- Provides a method for counterfactual inference such as "Would John have recovered from his illness if he had been given a certain treatment, given that he was not given the treatment and did not survive".

Examples:

    # Import RV and cGraph
    from because.causality import RV
    from because.causality import cGraph
    # Create a simple two-variable causal model
    A = RV(name='A')
    B = RV(name='B', parentNames=['A'])
    mod = [A, B]
    # Create a cGraph from a dataset and model
    # Dataset is same format as for ProbSpace above.
    cg = cGraph(rvList=mod, data=myABdataset)
    # Or create cGraph from a ProbSpace instance
    cg = cGraph(rvList=mod, ps=myPS)
    # Reverse the direction of the A->B link.
    cg.reverseLink('A', 'B')
    # Assess the model to see how consistent it is with the data.
    testResults = cg.testModel()
    # Perform an intervention.
    # Determine the effect on B of setting A to 1.
    # The resulting distribution shows the range of values
    # that B would take on if A were forced to 1.
    distr = cg.Intervene('B', [('A', 1)])
    # Find the Average Causal Effect (ACE) of A on B.
    effect = cg.ACE('A', 'B')

The Causal Subsystem additionally provides a Causal Discovery method known as _cDisc_.  cDisc is given a dataset or ProbSpace instance, and attempts to recover a Causal Model from the data.  The Causal Model is returned as a cGraph instance as described above.  From there, the model may be enhanced using cGraph's Model Editing capabilities, Tested for consistency with the data, or used directly for Caual Inference such as Intervention or Causal Metrics.

Examples:

    from because.causality import cdisc
    # Run a discovery on all variables in the dataset
    myCGraph = cdisc.discover(ps=myProbSpace)
    # Run a discovery on a subset of variables
    myCGraph = cdisc.discover(ps=myProbSpace, varNames=['A','B', 'C'])

### Command-Line Utilities

#### Assessing a synthetic model:

The command line program validationTest will utilize validationTest.py and a synthetic model with generated data to determine the extent to which the defined Causal Model is consistent with the generated data.

    python causality/test/validationTest.py <model file> [<numRecords>]

    Example:
        python causality/test/validationTest.py models/M3.py
        python causality/test/validationTest.py models/M3.py 10000


    If <numRecords> is not provided then all generated records will be used.
    If <numRecords> is provided, it will take a sample of <numRecords> from the generated data.  
    If <numRecords> is greater than or equal to the number of generated records, then all records will be used.

#### Causal Discovery:

Causal Discovery doesn't directly provide a command-line interface, but can be indirectly invoked using the _cModel_ command line from the Visualization subsystem.  That program uses cDisc to perform Causal Discovery on a provided dataset, and then produces a graphical output of the resulting Causal Model.  See Visualization / Causal Model above for details.

#### Other command-line programs:

_interventionTest.py_ provides a test and demonstration of the intervention logic, and causal metrics.  It is hard coded to test between variables 'A' and 'C', but can be modified locally to test any desired variables.  It demonstrates use of cGraph and  getData.

    python causality/test/interventionTest.py <model file>

    Example:
        python  causality/test/interventionTest.py models/M3.py



## References:

[1] Mackenzie, D. and Pearl, J. (2018) _The Book of Why._ Basic Books.

[2] Pearl, J. (2000, 2009) _Causality: Models, Reasoning_, and Inference.  Cambridge University Press

[3] Pearl, J., Glymour, M., Jewell, N. (2016) _Causal Inference In Statistics._ John Wiley and Sons.

[4] Morgan, S.L. and Winship, C. (2014) _Counterefactuals and Causal Inference: Methods and Principles for Social Research, Analytical Methods for Social Research._ 2nd edn.  Cambridge University Press.

[5] Shimizu, S. and Hyvarinen, A. (2006) _A linear non-Gaussian aclyclic model for causal discovery._ Journal of Machine Learning Research, 7:2003-2030.

[6] Spirites, P. and Glymour, C. (1991) _An algorithm for fast recovery of sparse causal graphs._ Computer Review, 9:67-72.


