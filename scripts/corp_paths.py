
class CorpPaths:

    sourceMapPath = None
    destMapPath = None
    ctfPath = None
    propsPath = None

    def __init__(self, sourceMapPath = None, destMapPath = None, ctfPath = None, propsPath = None):
        self.sourceMapPath = sourceMapPath
        self.destMapPath = destMapPath
        self.ctfPath = ctfPath
        self.propsPath = propsPath

    def getSourceMapPath(self):
        return self.sourceMapPath
    def getDestMapPath(self):
        return self.destMapPath
    def getCtfPath(self):
        return self.ctfPath
    def getPropsPath(self):
        return self.propsPath
