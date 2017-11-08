from enum import Enum
from collections import defaultdict
import itertools

from one2one import one2one
from ctf_writer import CTFFile

class Lang(Enum):
    SOURCE = 0
    DEST = 1

class Bicorpus:


    #########################RAW, INPUT TEXT########################
    rawSource = None
    rawDest = None

    #########################CORPUS TEXT#############################
    #The original (cleaned) lines of the text with no BPE merges
    #Also available via altLines[0]
    sourceLines = None
    destLines = None
    #Dictionary: int -> (list of source langauge lines, list of dest language lines)
    #The integer key indicates the number of byte-pair merges
    altLines = None
    



    

    #The original one-hot vocabulary indices with no BPE merges
    #Also available via altIndices[0]
    sourceMap = None
    destMap = None
    #Dictionary: int -> (Source word<-->int mapping, Dest word<-->int mapping)
    #The integer key indicates the number of byte-pair merges
    altMap= None

    #Only to be used in our earlier model, while still representing input as full words
    sourceWordCounts = None
    destWordCounts = None


    def __init__(self, sourceLines, destLines, maxWords = None, maxSequences = None):
        if len(sourceLines) != len(destLines):
            raise ValueError("Unequal number of source and destination language lines.")
        if maxSequences and (maxSequences > len(sourceLines)):
            raise ValueError("More sequences requested than available in sourceLines and destLines.")

        if not(maxSequences): maxSequences = len(sourceLines)
        self.rawSource = sourceLines
        self.rawDest = destLines


        self.sourceLines = []
        self.destLines = []

        self.sourceWordCounts = defaultdict(int)
        self.destWordCounts = defaultdict(int)
        self.sourceMap, self.destMap = self.__genIndexDicts(maxWords, maxSequences)

        self.altLines = {}
        self.altLines[0] = (self.sourceLines, self.destLines)

        self.altMap = {}
        self.altMap[0] = (self.sourceMap, self.destMap)


##########################Static Functions##################################

  ########################Class constants##########################
    @staticmethod
    def START():
        return "<s>"

    @staticmethod
    def END():
        return "</s>"


    @staticmethod
    def UNK():
        return "<UNK>"


    @staticmethod
    def __UNKIndex():
        return 0;

    @staticmethod
    def __EMPTY():
        return ""

    @staticmethod
    def __BADToken():
        return "<ERROR>"

   

   #######################Class methods##############################
    @staticmethod
    def __addAnnotativeTokens(wordMapping, nextIndex):
       wordMapping[Bicorpus.START()] = nextIndex
       nextIndex += 1
       wordMapping[Bicorpus.END()] = nextIndex

    @staticmethod
    def __cleanToken(token):
        chars = []
        for char in token:
            if char.isalpha(): chars.append(char.lower())
            elif char.isdigit(): return Bicorpus.__BADToken()  #Punctuation can be skipped over, but numeric data will just confuse the translation
    
        return Bicorpus.__EMPTY().join(chars)

    @staticmethod
    def __cleanSequence(sequence):
        tokens = [Bicorpus.__cleanToken(token) for token in sequence.split(" ")]
        return " ".join(tokens)
##########################Private functions#################################


    def __processToken(self, token, lang):
        if lang == Lang.SOURCE: self.sourceWordCounts[token] += 1
        else:                   self.destWordCounts[token] += 1
        
    def __processSequence(self, line, lang):
        lines = self.sourceLines if lang == Lang.SOURCE else self.destLines

        for token in line.split():
            self.__processToken(token, lang)

	#Add annotative tokens after counting word tokens
        lines.append( " ".join([Bicorpus.START(), line, Bicorpus.END()]) )

    def __processBisequence(self, readIndex):
        """
	    Return True if the sequences could be written and False otherwise.
	"""

        sourceLine = Bicorpus.__cleanSequence(self.rawSource[readIndex])
        destLine = Bicorpus.__cleanSequence(self.rawDest[readIndex])

        if (Bicorpus.__BADToken() in sourceLine) or (Bicorpus.__BADToken() in destLine): return False

        self.__processSequence(sourceLine, lang = Lang.SOURCE)
        self.__processSequence(destLine, lang = Lang.DEST)
        return True


    #Returns a tuple of (word to index dictionary, index to word dictionary)
    def __indexDict(self, lang, size):
       wordCounts = self.sourceWordCounts if lang == Lang.SOURCE else self.destWordCounts

       #Words in descending order of frequency
       words = sorted(wordCounts, key = wordCounts.get, reverse = True)

       wordMap = one2one(str, int, unknown_x = Bicorpus.UNK(), unknown_y = size)

       i = 0
       while i < size:
           wordMap[ words[i] ] = i
           i += 1

       Bicorpus.__addAnnotativeTokens(wordMap, i + 1) #Index i = size is already taken by unknown_y, so increment i
       return wordMap

    def __genIndexDicts(self, numWords, maxSequences):
        logFrequency = maxSequences // 20 if maxSequences > 20 else 1

        for i in range(maxSequences):
            self.__processBisequence(i)
            if (i + 1) % logFrequency == 0: print("{} sequences read.".format(i + 1), flush = True)

        #Ensure both source and destination vocabularies have equal vocabulary sizes to get CNTK to work (why is that necessary?)
        minVocabSize = min(len(self.sourceWordCounts), len(self.destWordCounts))
        if not(numWords) or (minVocabSize < numWords):
            numWords = minVocabSize

        sourceMap = self.__indexDict(Lang.SOURCE, numWords)
        destMap = self.__indexDict(Lang.DEST, numWords)

        return sourceMap, destMap

############################Public functions##############################


    def training_lines(self):
        return self.sourceLines, self.destLines

    def getMaps(self):
        return self.sourceMap, self.destMap

    def writeCTF(self, path, sourceLang, destLang):
        writer = CTFFile(path, sourceLang, destLang, self.sourceMap, self.destMap)
        for sourceLine, destLine in itertools.zip_longest(self.sourceLines, self.destLines, fillvalue = Bicorpus.__BADToken()):
           writer.writeSequence(sourceLine, destLine) 
        writer.close()

    def writeMapping(self, path, lang):
        wordMap = self.sourceMap if lang == Lang.SOURCE else self.destMap
        wordMap.write(path)
