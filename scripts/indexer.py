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
        self.sourceWordCounts = defaultdict(int)
        self.destWordCounts = defaultdict(int)

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
    
        return Indexer.__EMPTY().join(chars)

    def __indexDict(self, lang, size):
       wordCounts = self.sourceWordCounts if lang == Lang.SOURCE else self.destWordCounts
       if not(size): size = len(wordCounts) 

       #Words in descending order of frequency
       words = sorted(wordCounts, key = wordCounts.get, reverse = True)

       indices = {}
       i = 0
       for i in range(size):
           #print("Adding \"" + words[i] + "\" to " + str(lang) + " vocabulary.")
           indices[words[i]] = i

       return indices


    def __processToken(self, token, lang):
        cleaned = Indexer.__cleanToken(token)
        if lang == Lang.SOURCE: self.sourceWordCounts[cleaned] += 1
        else:                   self.destWordCounts[cleaned] += 1
        

    def __processSequence(self, line, lang):
        for token in line.split():
            self.__processToken(token, lang)

#####################Public Functions##################################
    
    def processBisequence(self, sourceLine, destLine):
        self.__processSequence(sourceLine, lang = Lang.SOURCE)
        self.__processSequence(destLine, lang = Lang.DEST)

    def indexDicts(self, sourceSize = None, destSize = None):
       return self.__indexDict(Lang.SOURCE, sourceSize), self.__indexDict(Lang.DEST, destSize)


