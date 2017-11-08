import one2one

class CTFFile:
    """
        Handles writing parallel lingual data to a CNTK Test (CTF) file
    """


    sequenceId = 0
    out = None      #CTF output file
    sourceLabel = ""
    destLabel = ""
    sourceMapping = None
    destMapping = None

    def __init__(self, path, sourceLabel, destLabel, sourceMapping = None, destMapping = None):
        self.out = open(path, "w")
        self.sourceLabel = sourceLabel
        self.destLabel = destLabel
        self.sourceMapping = sourceMapping
        self.destMapping = destMapping
        

    def writeSequence(self, sourceLine, destLine):
       sourceTokens = sourceLine.split()
       destTokens = destLine.split()

       i = 0
       while i < len(sourceTokens) and i < len(destTokens):
           if i > 0: self.out.write("\n")
           self.__writeRow(self, sourceTokens[i], destTokens[i])
           i += 1
    
       i = self.__writeHalfRows(sourceTokens, self.sourceLabel, i)
       i = self.__writeHalfRows(destTokens, self.destLabel, i)
    
       self.sequenceId += 1

    def __WriteRow(self, sourceToken, destToken, i):
        row = str(self.sequenceId) + " |" + self.sourceLabel + " " + word2Vec(sourceToken) + "#| " + sourceToken + " |" + self.destLabel + " " + word2Vec(destToken) + "#| " + destToken

    def __writeHalfRows(self, tokens, label, i):
        while i < len(tokens):
           token = tokens[i]
           row = str(self.sequenceId) + ' |' + label + " " + token + "\n"
           self.out.write(row)
           i += 1
        return i
       

    def close(self):
        self.out.close()
