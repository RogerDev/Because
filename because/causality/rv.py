"""
Random Variable Definition Module

Defines Class RV (Random Variable) as well as the Enumeration
for the variables data type.
"""
from enum import Enum

"""
RV Data Type enumeration.

Currently only NUMERIC and CATEGORICAL are supported.
BINARY variables should be treated as CATEGORICAL.

In most cases the default NUMERIC can be used.  The system will
automatically detect variables with low cardinality and treat them
as discrete.  Categorical variables with high cardinality (>100)
should be explicitly marked as CATEGORICAL.
"""
class RVType(Enum):
    NUMERIC = 1
    CATEGORICAL = 2

    """
    Random Variable Definition for causal modeling.

    Args:
        name(string) The name of the RV
        parentNames(list) [Optional]The list of causal parent names for this variable. Default empty list.
        isObserved(Boolean) [Optioal]True (default) if the variable is observed (i.e. has a data series).
        dataType(RVType) [Optional]Defined by RVType.  Current values are Numeric (default) or Categorical.
        forwardFunc(Function) Not currently used.
        backwardFunc(Function) Not currently used.
    """
class RV:
    def __init__(self, name, parentNames=[], isObserved=True, dataType=RVType.NUMERIC, forwardFunc=None, backwardFunc=None):
        self.name = name
        self.parentNames = parentNames
        self.isObserved = isObserved
        self.dataType = dataType
        self.forwardFunc = forwardFunc
        self.backwardFunc = backwardFunc


