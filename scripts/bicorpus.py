from enum import Enum
from collections import defaultdict
import itertools

from one2one import one2one

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


    def __init__(self, sourceLines, destLines, vocabSize = None, numSequences = None):
        if len(sourceLines) != len(destLines):
            raise ValueError("Unequal number of source and destination language lines.")
        if numSequences and (numSequences > len(sourceLines)):
            raise ValueError("More sequences requested than available in sourceLines and destLines.")

        if not(numSequences): numSequences = len(sourceLines)
        self.rawSource = sourceLines
        self.rawDest = destLines


        self.sourceLines = []
        self.destLines = []

        self.sourceWordCounts = defaultdict(int)
        self.destWordCounts = defaultdict(int)
        self.sourceMap, self.destMap = self.__genIndexDicts(vocabSize, numSequences)

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
    def UNKWord():
        return "<UNK>"


    @staticmethod
    def __UNKIndex():
        return 0;

    @staticmethod
    def __EMPTY():
        return ""

    @staticmethod
    def __BADToken():
        return "<BAD>"

   

   #######################Class methods##############################
    @staticmethod
    def __addAnnotativeTokens(wordMapping, nextIndex):
       wordMapping[Bicorpus.START()] = nextIndex
       nextIndex += 1
       wordMapping[Bicorpus.END()] = nextIndex
       nextIndex += 1
       wordMapping[Bicorpus.EMPTY()] = nextIndex

    @staticmethod
    def __cleanToken(token):
        chars = []
        for char in token:
            if char.isalpha(): chars.append(char.lower())
            elif char.isdigit(): return Bicorpus.__BAD()  #Punctuation can be skipped over, but numeric data will just confuse the translation
    
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
        lines.insert( " ".join([Bicorpus.start_token(), line, Bicorpus.end_token()]) )

    def __processBisequence(self, readIndex):
        """
	    Return True if the sequences could be written and False otherwise.
	"""

        sourceLine = Bicorpus.__cleanSequence(self.sourceLines[readIndex])
        destLine = Bicorpus.__cleanSequence(self.destLines[readIndex])

        if (Bicorpus.BADToken() in sourceLine) or (Bicorpus.BADToken() in destLine): return False

        self.__processSequence(sourceLine, lang = Lang.SOURCE)
        self.__processSequence(destLine, lang = Lang.DEST)
        return True


    #Returns a tuple of (word to index dictionary, index to word dictionary)
    def __indexDict(self, lang, size):
       wordCounts = self.sourceWordCounts if lang == Lang.SOURCE else self.destWordCounts

       #Words in descending order of frequency
       words = sorted(wordCounts, key = wordCounts.get, reverse = True)

       wordMap = one2one(unknown_x = Bicorpus.__UNKWord(), unknown_y = 0)
       i = 1
       while i < size + 1:
           wordMap[ words[i] ] = i
           i += 1

       Bicorpus.__addAnnotativeTokens(wordMap, i)
       return wordMap

    def __genIndexDicts(self, numWords, numSequences):
        logFrequency = numSequences // 20 if numSequences > 20 else 1

        for i in range(numSequences):
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

    def writeMapping(self, path, lang):
        wordMap = self.sourceMap if lang == Lang.SOURCE else self.destMap

	with open(path, "w") as dictFile:
            dictFile.write( str(len(wordMap)) + "\n") #Vocabulary size

            toWrite = "\n".join( [ word + " " + index for word, index in wordMap.items() ] )
            dictFile.write(toWrite)
	
	print("Wrote", path)


