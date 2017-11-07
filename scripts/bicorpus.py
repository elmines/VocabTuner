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
    altW2I = None
    altI2W = None

    #The original (cleaned) lines of the text with no BPE merges
    #Also available via altLines[0]
    sourceLines = None
    destLines = None

    #The original one-hot vocabulary indices with no BPE merges
    #Also available via altIndices[0]
    sourceW2I = None
    sourceI2W = None
    destW2I = None
    destI2W = None 

    #Only to be used in our earlier model, while still representing input as full words
    sourceWordCounts = None
    destWordCounts = None


    def __init__(self, sourceLines, destLines, vocabSize = None, numSequences = None):
        if len(sourceLines) != len(destLines):
            raise ValueError("Unequal number of source and destination language lines.")
        if numSequences and (numSequences > len(sourceLines)):
            raise ValueError("More sequences requested than available in sourceLines and destLines.")

        if not(numSequences): numSequences = len(sourceLines)
        self.sourceLines = sourceLines
        self.destLines = destLines
        
        self.sourceWordCounts = defaultdict(int)
        self.destWordCounts = defaultdict(int)
        (self.sourceW2I, self.destW2I), (self.sourceI2W, self.destI2W) = self.__genIndexDicts(vocabSize, numSequences)

        self.altLines = {}
        self.altLines[0] = (self.sourceLines, self.destLines)

        self.altW2I = {}
        self.altW2I[0] = (self.sourceW2I, self.destW2I)

        self.altI2W = {}
        self.altI2W[0] = (self.sourceI2W, self.destW2I)


##########################Static Functions##################################

  ########################Class constants##########################
    @staticmethod
    def start_token():
        return "<s>"

    @staticmethod
    def end_token():
        return "</s>"

    @staticmethod
    def __EMPTY():
        return ""

   #######################Class methods##############################

    @staticmethod
    def __cleanToken(token):
        chars = []
        for char in token:
            if char.isalpha(): chars.append(char.lower())
            elif char.isdigit(): chars.append(char)
    
        return Bicorpus.__EMPTY().join(chars)

##########################Private functions#################################

    @staticmethod
    def __cleanSequence(sequence, index = -1):
        tokens = [Bicorpus.__cleanToken(token) for token in sequence.split(" ")]
        #if index == 0: print(tokens)
        return " ".join(tokens)

    def __processToken(self, token, lang):
        if lang == Lang.SOURCE: self.sourceWordCounts[token] += 1
        else:                   self.destWordCounts[token] += 1
        
    def __processSequence(self, index, lang):
        lines = self.sourceLines if lang == Lang.SOURCE else self.destLines

        for token in lines[index].split():
            self.__processToken(token, lang)

	#Add annotative tokens after counting word tokens
        lines[index] = " ".join([Bicorpus.start_token(), lines[index], Bicorpus.end_token()])

    def __processBisequence(self, index):

        self.sourceLines[index] = Bicorpus.__cleanSequence(self.sourceLines[index])
        self.destLines[index] = Bicorpus.__cleanSequence(self.destLines[index])


        self.__processSequence(index, lang = Lang.SOURCE)
        self.__processSequence(index, lang = Lang.DEST)


    #Returns a tuple of (word to index dictionary, index to word dictionary)
    def __indexDicts(self, lang, size):
       wordCounts = self.sourceWordCounts if lang == Lang.SOURCE else self.destWordCounts
       if not(size) or size > len(wordCounts): size = len(wordCounts) 

       #Words in descending order of frequency
       words = sorted(wordCounts, key = wordCounts.get, reverse = True)

       w2i = {}
       i2w = {}
       for i in range(size):
           w2i[words[i]] = i
           i2w[i] = words[i]

       #Add annotative tokens
       w2i[Bicorpus.start_token()] = size
       w2i[Bicorpus.end_token()] = size + 1
        
       i2w[size] = Bicorpus.start_token()
       i2w[size + 1] = Bicorpus.end_token()

       return w2i, i2w

    def __genIndexDicts(self, vocabSize, numSequences ):
        logFrequency = numSequences // 20 if numSequences > 20 else 1
        for i in range(numSequences):
            self.__processBisequence(i)
            if (i + 1) % logFrequency == 0: print("{} sequences read.".format(i + 1), flush = True)

        #Ensure both source and destination vocabularies have equal vocabulary sizes to get CNTK to work
        minVocabSize = min(len(self.sourceWordCounts), len(self.destWordCounts))
        if not(vocabSize) or (minVocabSize < vocabSize):
            vocabSize = minVocabSize

        sourceW2I, sourceI2W = self.__indexDicts(Lang.SOURCE, vocabSize)
        destW2I, destI2W = self.__indexDicts(Lang.DEST, vocabSize)

        return (sourceW2I, destW2I), (sourceI2W, destI2W)

############################Public functions##############################


    def training_lines(self):
        return self.sourceLines, self.destLines

    def getW2IDicts(self):
        return self.sourceW2I, self.destW2I

    def getI2WDicts(self):
        return self.sourceI2W, self.destI2W

    def writeFiles(self):
