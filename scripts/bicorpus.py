from enum import Enum
from collections import defaultdict
import itertools

class Lang(Enum):
    SOURCE = 0
    DEST = 1

class Bicorpus:

    #Dictionary: int -> (list of source langauge lines, list of dest language lines)
    #The integer key indicates the number of byte-pair merges
    altLines = None
    
    #Dictionary: int -> (list of source langauge lines, list of dest language lines)
    #The integer key indicates the number of byte-pair merges
    altIndices = None

    #The original (cleaned) lines of the text with no BPE merges
    #Also available via altLines[0]
    sourceLines = None
    destLines = None

    #The original one-hot vocabulary indices with no BPE merges
    #Also available via altIndices[0]
    sourceIndices = None
    destIndices = None

    #Only to be used in our earlier model, while still representing input as full words
    sourceWordCounts = None
    destWordCounts = None


    def __init__(self, sourceLines, destLines, vocabSize = 30000, numSequences = None):
        if len(sourceLines) != len(destLines):
            raise ValueError("Unequal number of source and destination language lines.")
        if numSequences and (numSequences > len(sourceLines)):
            raise ValueError("More sequences requested than in file.")

        if not(numSequences): numSequences = len(sourceLines)
        self.sourceLines = sourceLines
        self.destLines = destLines
        
        self.sourceWordCounts = defaultdict(int)
        self.destWordCounts = defaultdict(int)
        self.sourceIndices, self.destIndices = self.__indexDicts(vocabSize, numSequences)

        self.altLines = {}
        self.altLines[0] = (self.sourceLines, self.destLines)

        self.altIndices = {}
        self.altIndices[0] = (self.sourceIndices, self.destIndices)




##########################Private functions#################################
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
    
        return Bicorpus.__EMPTY().join(chars)

    @staticmethod
    def __cleanSequence(sequence, index = -1):
        tokens = [Bicorpus.__cleanToken(token) for token in sequence.split(" ")]
        if index == 0: print(tokens)
        return " ".join(tokens)

    def __processToken(self, token, lang):
        if lang == Lang.SOURCE: self.sourceWordCounts[token] += 1
        else:                   self.destWordCounts[token] += 1
        
    def __processSequence(self, line, lang):
        for token in Bicorpus.__cleanSequence(line).split():
            self.__processToken(token, lang)

    def __processBisequence(self, index):
        if index == 0:
            print(self.sourceLines[index])
            print(self.destLines[index])

        self.sourceLines[index] = Bicorpus.__cleanSequence(self.sourceLines[index], index)
        self.destLines[index] = Bicorpus.__cleanSequence(self.destLines[index])

        if index == 0:
            print(self.sourceLines[index])
            print(self.destLines[index])

        self.__processSequence(self.sourceLines[index], lang = Lang.SOURCE)
        self.__processSequence(self.destLines[index], lang = Lang.DEST)

    def __indexDict(self, lang, size):
       wordCounts = self.sourceWordCounts if lang == Lang.SOURCE else self.destWordCounts
       if not(size) or size > len(wordCounts): size = len(wordCounts) 

       #Words in descending order of frequency
       words = sorted(wordCounts, key = wordCounts.get, reverse = True)

       indices = {}
       i = 0
       for i in range(size):
           #print("Adding \"" + words[i] + "\" to " + str(lang) + " vocabulary.")
           indices[words[i]] = i

       return indices

    def __indexDicts(self, vocabSize, numSequences):
        logFrequency = numSequences // 20
        for i in range(numSequences):
            self.__processBisequence(i)
            if (i + 1) % logFrequency == 0: print("{} sequences read.".format(i + 1))

        sourceIndices = self.__indexDict(Lang.SOURCE, vocabSize)
        destIndices = self.__indexDict(Lang.DEST, vocabSize)

        return sourceIndices, destIndices

############################Public functions##############################

    def training_lines(self):
        return self.sourceLines, self.destLines
