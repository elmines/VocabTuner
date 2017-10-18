from collections import defaultdict
from enum import Enum


class Lang(Enum):
    SOURCE = 0
    DEST = 1

class Indexer:
    """
        Produces indices for one-hot vector representations of words
    """


    sourceWordCounts = None
    destWordCounts = None

    def __init__(self):
        sourceWordCounts = defaultdict(int)
        destWordCounts = defaultdict(int)

########################Private Functions###############################

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
    
        return EMPTY().join(chars)

    def __indexDict(self, lang, size):
       wordCounts = sourceWordCounts if lang == Lang.SOURCE else destWordCounts
       if not(size): size = len(wordCounts) 

       #Words in descending order of frequency
       words = sorted(wordCounts, key = wordCounts.get, reverse = True).keys()

       indices = {}
       i = 0
       for i in range(size):
           indices[words[i]] = i

       return indices


    def __processToken(self, token, lang):
        cleaned = Indexer.__cleanToken(token)
        if lang == Lang.SOURCE: sourceWordCounts[cleaned] += 1
        else:                   destWordCounts[cleaned] += 1
        

    def __processSequence(self, line, lang):
        for token in line.split():
            self.__processToken(token, lang)

#####################Public Functions##################################
    
    def processBisequence(self, sourceLine, destLine):
        self.__processSequence(sourceLine, lang = Lang.SOURCE)
        self.__processSequence(destLine, lang = Lang.DEST)

    def indexDicts(self, sourceSize = None, destSize = None):
       return self.__indexDict(Lang.SOURCE, sourceSize), self.__indexDict(Lang.DEST, destSize)


