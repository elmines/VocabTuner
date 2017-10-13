numSentences = 25
sourceLangPath = "../corpora/europarl-v7.es-en.es"
destLangPath = "../corpora/europarl-v7.es-en.en"
ctfPath = "../data/euroParles-en.ctf"

sourceColumn = "S0"
destColumn = "S1"
EMPTY = ""

source = open(sourceLangPath, "r")
dest = open(destLangPath, "r")
ctf = open(ctfPath, "w")

def cleanToken(token):
    chars = []
    for char in token:
        if char.isalpha(): chars.append(char.lower())
        elif char.isdigit(): chars.append(char)

    if len(chars) > 0: return EMPTY.join(chars)
    else:              return EMPTY               #Case in which we have just non alphanumeric characters by themselves?

def writeSequence(sourceLine, destLine, seqId):
   sourceTokens = sourceLine.split()
   destTokens = destLine.split()

   i = 0
   while i < len(sourceTokens) and i < len(destTokens):
       sourceToken = cleanToken(sourceTokens[i])
       destToken = cleanToken(destTokens[i])

       row = str(seqId) + ' |' + sourceColumn + " " + sourceToken + " |" + destColumn + " " + destToken + "\n"
       ctf.write(row)

       i += 1

   while i < len(sourceTokens):
       sourceToken = cleanToken(sourceTokens[i])
       row = str(seqId) + ' |' + sourceColumn + " " + sourceToken + "\n"
       ctf.write(row)
       i += 1

   while i < len(destTokens):
       destToken = cleanToken(destTokens[i])
       row = str(seqId) + ' |' + destColumn + " " + destToken + "\n"
       ctf.write(row)
       i += 1


sourceLine = source.readline()
destLine = dest.readline()

seqId = 0
while (seqId < numSentences) and sourceLine and destLine:
    writeSequence(sourceLine, destLine, seqId)
    seqId += 1 
    sourceLine = source.readline()
    destLine = dest.readline()

source.close()
dest.close()
ctf.close()
