from enum import Enum
from collections import defaultdict
import os

#Custom modules
from one2one import one2one
from ctf_writer import CTFFile

class Lang(Enum):
    SOURCE = 0
    DEST = 1

class Bicorpus:

    #########################CORPUS TEXT#############################
    rawSource = None
    rawDest = None


    #The lines of the text with annotations.
    sourceLines = None
    destLines = None

    ########################PROPS VARIABLES##########################
    sourceTokensCount = 0
    destTokensCount = 0

    #The original one-hot vocabulary indices with no BPE merges
    sourceMap = None
    destMap = None

    #Only to be used in our earlier model, while still representing input as full words
    sourceWordCounts = None
    destWordCounts = None

    #Paths for files related to the Bicorpus
    sourceMapPath = None
    destMapPath = None
    ctfPath = None


    def __init__(self, sourceLines, destLines, maxWords = None, maxSequences = None):
        if len(sourceLines) != len(destLines):
            raise ValueError("Unequal number of source and destination language lines.")
        if maxSequences and (maxSequences > len(sourceLines)):
            raise ValueError("More sequences requested than available in sourceLines and destLines.")

        if not(maxSequences): maxSequences = len(sourceLines)
        self.rawSource = sourceLines
        self.rawDest = destLines
        self.sourceLines = None
        self.destLines = None

        self.sourceWordCounts = defaultdict(int)
        self.destWordCounts = defaultdict(int)

        self.sourceMapPath = None
        self.destMapPath = None
        self.ctfPath = None


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


    #@staticmethod
    #def __UNKIndex():
        #return 0;

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


    def __processStrLine(self, line, lang):
       tokenCount = 0
       for token in tokens:
           self.__processToken(token, lang)
           tokenCount += 1

        tokenCount += 2 #For start and end sequences
        if lang == Lang.SOURCE: self.sourceTokensCount += tokenCount
        else:                   self.destTokensCount += tokenCount

        return " ".join([Bicorpus.START(), line, Bicorpus.END()])

    def __processListLine(self, line, lang):

        tokenCount = 0

        for word in line:
            for character in word:
                self.__processToken(token, lang)
                tokenCount += 1

        tokenCount += 2
        if lang == Lang.SOURCE: self.sourceTokensCount += tokenCount
        else:                   self.destTokensCount += tokenCount

        return [Bicorpus.START()] + line + [Bicorpus.END()]
        
    def __processSequence(self, line, lang):
        lines = self.sourceLines if lang == Lang.SOURCE else self.destLines

        tokenCount = 0

        if type(line) == list: processed = self.__processListLine(line, lang)
        else:                  processed = self.__processStrLine(line, lang)

        lines.append


    def __processBisequence(self, readIndex):
        """
	    Return True if the sequences could be written and False otherwise.
	"""

        #sourceLine = Bicorpus.__cleanSequence(self.rawSource[readIndex])
        #destLine = Bicorpus.__cleanSequence(self.rawDest[readIndex])

        sourceLine = self.rawSource[readIndex]
        destLine = self.rawDest[readIndex]

        #if (Bicorpus.__BADToken() in sourceLine) or (Bicorpus.__BADToken() in destLine): return False

        self.sourceLines.append( self.__processSequence(sourceLine, Lang.SOURCE) )
        self.destLines.append( self.__processSequence(destLine, Lang.DEST) )
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
            if (i + 1) % logFrequency == 0:
                print("{} sequences read.".format(i + 1), flush = True)

        #Ensure both source and destination vocabularies have equal vocabulary sizes to get CNTK to work (why is that necessary?)
        minVocabSize = min(len(self.sourceWordCounts), len(self.destWordCounts))
        if not(numWords) or (minVocabSize < numWords):
            numWords = minVocabSize

        sourceMap = self.__indexDict(Lang.SOURCE, numWords)
        destMap = self.__indexDict(Lang.DEST, numWords)

        return sourceMap, destMap

############################Public functions##############################


    def getTrainingLines(self):
        return self.sourceLines, self.destLines



    
    def getMaps(self):
        """
        Returns the vocabulary mappings, generating them if need be.
        """
        if not(self.sourceMap):
            self.sourceMap, self.destMap = self.__genIndexDicts(maxWords, maxSequences)
            self.altMap[0] = (self.sourceMap, self.destMap)

        return self.sourceMap, self.destMap

    def getMapPath(self, lang):
        return self.sourceMapPath if lang == Lang.SOURCE else self.destMapPath

    def getSourceMapPath(self):
        return self.getMapPath(Lang.SOURCE)

    def getDestMapPath(self):
        return self.getMapPath(Lang.DEST)

    def getCTFPath(self):
        return self.ctfPath

    def writeCTF(self, path, sourceLang, destLang):
        """
	Writes sourceLines and destLines to a CTF file at an absolute path.
	"""
        writer = CTFFile(path, sourceLang, destLang, self.sourceMap, self.destMap)
        writer.writeSequences(self.sourceLines, self.destLines)
        writer.close()
        self.ctfPath = path

    def writeMapping(self, path, lang):
        """
	Writes one of the mappings (determined by lang) to an absolute path.
	"""
        wordMap = self.sourceMap if lang == Lang.SOURCE else self.destMap
        wordMap.write(path)
        if lang == Lang.SOURCE: self.sourceMapPath = path
        else:                   self.destMapPath   = path

    def writeProps(self, path):
        """
        Writes various propertiets associated with the text (# sequences, #tokens, etc.)
        """
        with open(path, "w") as propsFile:
            propsFile.write( str( len(self.sourceLines) ) + "\n")
            num_tokens = str( max(self.sourceTokensCount, self.destTokensCount) )
            propsFile.write( num_tokens + "\n" ) 

