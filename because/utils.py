
def vprint(level, verbosity, *args):
    """
    Print only if verbosity is >= level.  Insert 2 spaces before
    printing for each level-1.
    """
    if verbosity >= level:
        strargs = [str(arg) for arg in args]
        outStr = '  ' * (level - 1) + ' '.join(strargs)
        print(outStr)
