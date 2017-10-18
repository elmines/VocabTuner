#import ctf_writer
from indexer import Indexer

numSentences = 100
vocabSize = 25
sourceLangPath = "../corpora/europarl-v7.es-en.es"
destLangPath = "../corpora/europarl-v7.es-en.en"

#ctfPath = "../data/euroParles-en.ctf"

#sourceColumn = "S0"
#destColumn = "S1"

source = open(sourceLangPath, "r")
dest = open(destLangPath, "r")
#ctf = ctf_writer.CTFFile(ctfPath, sourceColumn, destColumn)
ind = Indexer()


sourceLine = source.readline()
destLine = dest.readline()

count = 0
while (count < numSentences) and sourceLine and destLine:
    #ctf.writeSequence(sourceLine, destLine)

    ind.processBisequence(sourceLine, destLine)

    sourceLine = source.readline()
    destLine = dest.readline()
    count += 1

source.close()
dest.close()
#ctf.close()

sourceIndices, destIndices = ind.indexDicts(sourceSize = vocabSize, destSize = vocabSize)

print(sourceIndices)
print()
print(destIndices)

