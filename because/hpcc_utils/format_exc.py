"""
Common way to format exceptions for HPCC embedded functions

    Args:
        func(string): The name of the function in which the exception occurred.

    Returns:
        string: A formatted exception string.
"""
def format(func=''):
    import traceback as tb
    exc = tb.format_exc(limit=2)
    if len(exc) < 100000:
        return func + ': ' + exc
    else:
        return func + ': ' + exc[:200] + ' ... ' + exc[-200:]
