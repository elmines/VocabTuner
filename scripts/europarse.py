import os
import sys
import bicorpus
import corp_paths

import my_io

def __corpDir():
    return "../corpora"

def __docPrefix(sourceLang, destLang):
    return "europarl-v7." + sourceLang + "-" + destLang + "."

def __propsPath(sourceLang, destLang):
    return my_io.abs_path( os.path.join(__corpDir(), __docPrefix(sourceLang, destLang) + "props"), __file__)

def __ctfPath(sourceLang, destLang):
    return my_io.abs_path( os.path.join(__corpDir(), __docPrefix(sourceLang, destLang) + "ctf"), __file__ )

def __dataPath(sourceLang, destLang, lang):
    extension = sourceLang if lang == bicorpus.Lang.SOURCE else destLang
    return my_io.abs_path( os.path.join(__corpDir(), __docPrefix(sourceLang, destLang) + extension), __file__ )

def __dictPath(sourceLang, destLang, lang):
    extension = sourceLang if lang == bicorpus.Lang.SOURCE else destLang
    return my_io.abs_path( os.path.join(__corpDir(), __docPrefix(sourceLang, destLang) + extension + ".dict"), __file__ )


def parseCorpora(sourceLang, destLang, maxWords = 30000, maxSequences = 50000):
    """
    Returns a tuple of the generated Bicorpus and the paths to the language's
    one2one vector mappings and the CTF data file.
    """

    #cleanedPrefix = "cleaned-"
    #docPrefix = "europarl-v7." + sourceLang + "-" + destLang + "."
    #corpDir = "../corpora"

    #sourcePath = my_io.abs_path( os.path.join(corpDir, docPrefix + sourceLang), __file__ )
    #destPath = my_io.abs_path( os.path.join(corpDir, docPrefix + destLang), __file__ )
    sourcePath= __dataPath(sourceLang, destLang, bicorpus.Lang.SOURCE)
    destPath = __dataPath(sourceLang, destLang, bicorpus.Lang.DEST)

    print("Reading lines from", sourcePath)
    with open(sourcePath, "r", encoding = "utf-8") as sourceFile:
        sourceLines = sourceFile.readlines()
    
    print("Reading lines from", destPath)
    with open(destPath, "r", encoding = "utf-8") as destFile:
        destLines = destFile.readlines()
    
    bp = bicorpus.Bicorpus(sourceLines, destLines, maxWords = maxWords, maxSequences = maxSequences)


    #sourceMapPath = my_io.abs_path( os.path.join(corpDir, docPrefix + sourceLang + ".dict"), __file__ )
    #destMapPath = my_io.abs_path( os.path.join(corpDir, docPrefix + destLang + ".dict"), __file__ )

    sourceMapPath = __dictPath(sourceLang, destLang, bicorpus.Lang.SOURCE)
    destMapPath = __dictPath(sourceLang, destLang, bicorpus.Lang.DEST)

    bp.writeMapping(sourceMapPath, bicorpus.Lang.SOURCE)
    print("Wrote", sourceMapPath)

    bp.writeMapping(destMapPath, bicorpus.Lang.DEST) 
    print("Wrote", destMapPath)

    #ctfPath = my_io.abs_path( os.path.join(corpDir, docPrefix + "ctf"), __file__ )
    ctfPath = __ctfPath(sourceLang, destLang)
    bp.writeCTF(ctfPath, sourceLang, destLang)
    print("Wrote", ctfPath)

    propsPath = __propsPath(sourceLang, destLang)
    bp.writeProps(propsPath)
    print("Wrote", propsPath)

    #All the paths, exce are stored in the Bicorpus object
    return corp_paths.CorpPaths(sourceMapPath = sourceMapPath,
                                  destMapPath = destMapPath,
                                      ctfPath = ctfPath,
                                    propsPath = propsPath)

#Command-line interface
#if len(sys.argv) > 1:
    #langPair = sys.argv[1].split("-")
    #sourceLang, destLang = langPair[0], langPair[1]
    #parseCorpora(sourceLang, destLang)


def getPaths(sourceLang, destLang):
    return corp_paths.CorpPaths(sourceMapPath = __dictPath(sourceLang, destLang, bicorpus.Lang.SOURCE),
                                  destMapPath = __dictPath(sourceLang, destLang, bicorpus.Lang.DEST),
                                      ctfPath = __ctfPath(sourceLang, destLang),
                                    propsPath = __propsPath(sourceLang, destLang) )
