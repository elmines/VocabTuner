import sys
import europarse

def preprocessEuroparl(sourceLang, destLang, maxWords, maxSequences):
    bp = europarse.parseCorpora(sourceLang, destLang, maxWords = maxWords, maxSequences = maxSequences)
    return bp

def getEuroparlPaths(sourceLang, destLang):
    return europarse.getPaths(sourceLang, destLang)  


if len(sys.argv) > 1:
    i = 1
    langs = sys.argv[i].split("-")
    i += 1

    maxSequences = int(sys.argv[i]) if i < len(sys.argv) else 50000
    i += 1

    maxWords = int(sys.argv[i]) if i < len(sys.argv) else 30000

    preprocessEuroparl(langs[0], langs[1], maxWords, maxSequences)
