import one2one

class CTFFile:
    """
        Handles writing parallel lingual data to a CNTK Text (CTF) file
    """


    sequenceId = 0
    out = None      #CTF output file
    sourceLabel = ""
    destLabel = ""
    sourceMapping = None
    destMapping = None

    def __init__(self, path, sourceLabel, destLabel, sourceMapping, destMapping):
        self.out = open(path, "w")
        self.sourceLabel = sourceLabel
        self.destLabel = destLabel
        self.sourceMapping = sourceMapping
        self.destMapping = destMapping
        

    def writeSequence(self, sourceLine, destLine):
       sourceTokens = sourceLine.split() if type(sourceLine) == str else sourceLine
       destTokens = destLine.split() if type(destLine) == str else destLine

       i = 0
       while i < len(sourceTokens) and i < len(destTokens):
           self.__writeRow(sourceTokens[i], destTokens[i])
           i += 1
    
       i = self.__writeHalfRows(sourceTokens, self.sourceLabel, self.sourceMapping, i)
       i = self.__writeHalfRows(destTokens, self.destLabel, self.destMapping, i)
    
       self.sequenceId += 1

    def __writeRow(self, sourceToken, destToken):
        row = str(self.sequenceId) + " |" + self.sourceLabel + " " + str(self.sourceMapping[sourceToken]) + ":1 #|" + sourceToken + " |" + self.destLabel + " " + str(self.destMapping[destToken]) + ":1 #|" + destToken + "\n"
        self.out.write(row)

    def __writeHalfRows(self, tokens, label, mapping, i):
        while i < len(tokens):
           token = tokens[i]
           row = str(self.sequenceId) + ' |' + label + " " + str(mapping[token]) + ":1 #| " + token + "\n"
           self.out.write(row)
           i += 1
        return i
       

    def close(self):
        self.out.close()
