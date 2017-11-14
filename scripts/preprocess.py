import sys
import europarse

def preprocessEuroparl(sourceLang, destLang):
    bp = europarse.parseCorpora(sourceLang, destLang)
    return bp

def getEuroparlPaths(sourceLang, destLang):
    return europarse.getPaths(sourceLang, destLang)  

if len(sys.argv) == 2:
    #print("Called first branch.", flush = True)
    langs = sys.argv[1].split("-")
    preprocessEuroparl(langs[0], langs[1])
elif len(sys.argv) == 3:
    #print("Called second branch.", flush = True)
    preprocessEuroparl(sys.argv[1], sys.argv[2])
