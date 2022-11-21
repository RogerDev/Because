"""
A collection of utility functions that are used by various modules.
"""

def getCombos(inItems, maxRank=None, minRank=None):
    """
    Return all sets of inItems of size in [minRank, maxRank].
    minRank defaults to 1 (i.e. set size >= 1).
    maxRank defaults to the size of the inItems set.
    inItems can be strings or numbers.
    Sets are returned as a list of combos.  Each combo is a
    list of items from inItems without duplicates.
    """
    if maxRank is None:
        # Use the length of the input set as the maxRank by default
        maxRank = len(inItems)
    if minRank is None:
        # Use 1 as the default minRank.  The null set will not be
        # returned unless minRank is explicitly set to 0.
        minRank = 1
    levelItems = [] # List of levels, with a list of items for each level
    for l in range(0, maxRank+1):
        # Initialize a list for each level, including 0
        levelItems.append([])
    for l in range(1, maxRank+1):
        if l == 1:
            for j in range(len(inItems)):
                levelItems[l].append([j])
        else:
            prevLevel = levelItems[l-1]
            for i in range(len(prevLevel)):
                prevItem = prevLevel[i]
                prevEnd = prevItem[-1]
                for j in range(prevEnd+1, len(inItems)):
                    levelItems[l].append(prevItem + [j])
    outItems = []
    for l in range(minRank, maxRank+1):
        levItems = levelItems[l]
        if l == 0 and not levItems:
            outItems.append([])
        else:
            for item in levItems:
                itemVars = [inItems[j] for j in item]
                outItems.append(itemVars)
    return outItems
