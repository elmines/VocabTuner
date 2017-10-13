numSentences = 50000
sourceLangPath = "../corpora/europarl-v7.es-en.es"
destLangPath = "../corpora/europarl-v7.es-en.en"
ctfPath = "../data/euroParles-en.ctf"

source = open(sourceLangPath, "r")
dest = open(destLangPath, "r")
ctfPath = open(ctfPath, "w")

sourceLine = source.readline()
destLine = dest.readline()

while sourceLine and destLine:
    
