import os
import sys
import bicorpus

#Command-line interface
if len(sys.argv) > 1:
    langPair = sys.argv[1].split("-")
    sourceLang, destLang = langPair[0], langPair[1]
    parseCorpora(sourceLang, destLang)

def parseCorpora(sourceLang, destLang, numWords = 30000, numSequences = 50000):

    cleanedPrefix = "cleaned-"
    docPrefix = "europarl-v7." + sourceLang + "-" + destLang + "."
    corpDir = "../corpora"
    
    sourcePath = os.path.abspath( os.path.join(corpDir, docPrefix + sourceLang) )
    destPath = os.path.abspath( os.path.join(corpDir, docPrefix + destLang) )
    
    with open(sourcePath, "r", encoding = "utf-8") as sourceFile:
        print("Reading lines from", sourcePath)
        sourceLines = sourceFile.readlines()
    
    with open(destPath, "r", encoding = "utf-8") as destFile:
        print("Reading lines from", destPath)
        destLines = destFile.readlines()
    
    bp = bicorpus.Bicorpus(sourceLines, destLines, maxWords = numWords, maxSequences = numSequences)

    sourceMapPath = os.path.abspath( os.path.join(corpDir, docPrefix + sourceLang + ".dict") )
    destMapPath = os.path.abspath( os.path.join(corpDir, docPrefix + destLang + ".dict") )


    bp.writeMapping(sourceMapPath, bicorpus.Lang.SOURCE)
    print("Wrote", sourceMapPath)

    bp.writeMapping(destMapPath, bicorpus.Lang.DEST) 
    print("Wrote", destMapPath)

    ctfPath = os.path.abspath( os.path.join(corpDir, docPrefix + "ctf") )
    bp.writeCTF(ctfPath, sourceLang, destLang)
    print("Wrote", ctfPath)

    return (sourceMapPath, destMapPath, ctfPath)
