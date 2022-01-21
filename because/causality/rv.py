class RV:
    def __init__(self, name, parentNames, isObserved, dataType, forwardFunc, backwardFunc):
        self.name = name
        self.parentNames = parentNames
        self.isObserved = isObserved
        self.dataType = dataType
        self.forwardFunc = forwardFunc
        self.backwardFunc = backwardFunc
        self.saveParents = []


