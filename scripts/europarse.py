import os
import sys
import bicorpus

import my_io

def parseCorpora(sourceLang, destLang, maxWords = 30000, maxSequences = 50000):
    """
    Returns a tuple of the generated Bicorpus and the paths to the language's
    one2one vector mappings and the CTF data file.
    """

    cleanedPrefix = "cleaned-"
    docPrefix = "europarl-v7." + sourceLang + "-" + destLang + "."
    corpDir = "../corpora"

    sourcePath = my_io.abs_path( os.path.join(corpDir, docPrefix + sourceLang), __file__ )
    destPath = my_io.abs_path( os.path.join(corpDir, docPrefix + destLang), __file__ )

    with open(sourcePath, "r", encoding = "utf-8") as sourceFile:
        print("Reading lines from", sourcePath)
        sourceLines = sourceFile.readlines()
    
    with open(destPath, "r", encoding = "utf-8") as destFile:
        print("Reading lines from", destPath)
        destLines = destFile.readlines()
    
    bp = bicorpus.Bicorpus(sourceLines, destLines, maxWords = maxWords, maxSequences = maxSequences)

    sourcePath = my_io.abs_path( os.path.join(corpDir, docPrefix + sourceLang), __file__ )

    sourceMapPath = my_io.abs_path( os.path.join(corpDir, docPrefix + sourceLang + ".dict"), __file__ )
    destMapPath = my_io.abs_path( os.path.join(corpDir, docPrefix + destLang + ".dict"), __file__ )


    bp.writeMapping(sourceMapPath, bicorpus.Lang.SOURCE)
    print("Wrote", sourceMapPath)

    bp.writeMapping(destMapPath, bicorpus.Lang.DEST) 
    print("Wrote", destMapPath)

    ctfPath = my_io.abs_path( os.path.join(corpDir, docPrefix + "ctf"), __file__ )
    bp.writeCTF(ctfPath, sourceLang, destLang)
    print("Wrote", ctfPath)

    return (bp, sourceMapPath, destMapPath, ctfPath)

#Command-line interface
if len(sys.argv) > 1:
    langPair = sys.argv[1].split("-")
    sourceLang, destLang = langPair[0], langPair[1]
    parseCorpora(sourceLang, destLang)

