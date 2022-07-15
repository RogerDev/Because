from math import *
from numpy.random import *

def truncated(distSpec, lower=None, upper=None):
    """
    Truncate a distribution at a minimum value, a maximum
    value, or both.  Truncation is achieved by rejecting
    and regenerating any sample that falls outside the specified ranges.
    distSpec is a string with a distribution call for
    any distribution in numpy.random (e.g. 'normal(0,1)', 'exponential()')
    Returns a single sample.
    """
    while True:
        s = eval(distSpec)
        if (lower is None or s >= lower) and (upper is None or s <= upper):
            return s