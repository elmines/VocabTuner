import sys
import bicorpus

#Command-line interface
if len(sys.argv) > 1:
    langPair = sys.argv[1].split("-")
    sourceLang, destLang = langPair[0], langPair[1]
    parseCorpora(sourceLang, destLang)

def parseCorpora(sourceLang, destLang, numWords = 30000, numSequences = 50000):

    cleanedPrefix = "cleaned-"
    docPrefix = "europarl-v7."
    corpDir = "../corpora"
    
    sourcePath = "".join([corpDir, "/", docPrefix, sourceLang, "-", destLang, ".", sourceLang])
    destPath = "".join([corpDir, "/", docPrefix, sourceLang, "-", destLang, ".", destLang])
    
    
    with open(sourcePath, "r", encoding = "utf-8") as sourceFile:
        print("Reading lines from", sourcePath)
        sourceLines = sourceFile.readlines()
    
    with open(destPath, "r", encoding = "utf-8") as destFile:
        print("Reading lines from", destPath)
        destLines = destFile.readlines()
    
    bp = bicorpus.Bicorpus(sourceLines, destLines, maxWords = numWords, maxSequences = numSequences)

    sourceMapPath = "".join([corpDir, "/", docPrefix, sourceLang, "-", destLang, ".", sourceLang, ".dict"])
    destMapPath = "".join([corpDir, "/", docPrefix, sourceLang, "-", destLang, ".", destLang, ".dict"])

    bp.writeMapping(sourceMapPath, bicorpus.Lang.SOURCE)
    bp.writeMapping(destMapPath, bicorpus.Lang.DEST) 

    
    
    #ctfPath = "".join([corpDir, "/", cleanedPrefix, sourceLang, "-" destLang, ".ctf"])
    #bp.writeCTF(ctfPath)
