"""
Common way to format exceptions for HPCC embedded functions

    Args:
        func(string): The name of the function in which the exception occurred.

    Returns:
        string: A formatted exception string.
"""
def format(func=''):
    import traceback as tb
    import sys
    e_t, e, tback = sys.exc_info()
    if e_t.__name__ == 'AssertionError':
        outStr =  func + ': Error -- ' + e.args[0]
    else:
        exc = tb.format_exc()
        if len(exc) < 900:
            outStr = func + ': ' + exc
        else:
            outStr = func + ': ' + exc[:400] + ' ... ' + exc[-400:]
    return outStr
