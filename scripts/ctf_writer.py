
class CTFFile:
    """
        Handles writing parallel lingual data to a CNTK Test (CTF) file
    """


    sequenceId = 0
    out = None      #CTF output file
    sourceLabel = ""
    destLabel = ""

    def __init__(self, path, sourceLabel, destLabel):
        self.out = open(path, "w")
        self.sourceLabel = sourceLabel
        self.destLabel = destLabel
        
    #Class constant
    @staticmethod
    def __EMPTY():
        return ""

    @staticmethod
    def __cleanToken(token):
        chars = []
        for char in token:
            if char.isalpha(): chars.append(char.lower())
            elif char.isdigit(): chars.append(char)
    
        if len(chars) > 0: return CTFFile.__EMPTY().join(chars)
        else:              return CTFFile.__EMPTY()               #Case in which we have just non alphanumeric characters by themselves?

    
    def writeSequence(self, sourceLine, destLine):
       sourceTokens = sourceLine.split()
       destTokens = destLine.split()

       i = 0
       while i < len(sourceTokens) and i < len(destTokens):
           sourceToken = CTFFile.__cleanToken(sourceTokens[i])
           destToken = CTFFile.__cleanToken(destTokens[i])
           row = str(self.sequenceId) + ' |' + self.sourceLabel + " " + sourceToken + " |" + self.destLabel + " " + destToken + "\n"
           self.out.write(row)
    
           i += 1
    
       i = self.__writeHalfRows(sourceTokens, self.sourceLabel, i)
       i = self.__writeHalfRows(destTokens, self.destLabel, i)
    
       self.sequenceId += 1


    def __writeHalfRows(self, tokens, label, i):
        while i < len(tokens):
           token = CTFFile.__cleanToken(tokens[i])
           row = str(self.sequenceId) + ' |' + label + " " + token + "\n"
           self.out.write(row)
           i += 1
        return i
       

    def close(self):
        self.out.close()
