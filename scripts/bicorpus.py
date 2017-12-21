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

    maxSequences = 0

    ########################PROPS VARIABLES##########################
    sourceTokensCount = 0
    destTokensCount = 0

    #######################VOCABULARY VARIABLES#####################
    #One-hot indices
    sourceMap = None
    destMap = None
    sourceWordCounts = None
    destWordCounts = None
    maxWords = 0

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
        self.sourceLines = []
        self.destLines = []
        self.maxSequences = maxSequences

        self.sourceWordCounts = defaultdict(int)
        self.destWordCounts = defaultdict(int)
        self.maxWords = maxWords

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
        """
        Returns True for a good, clean token and False otherwise.
        """
        cleaned = Bicorpus.__cleanToken(token)

        #If there was actually any text left after cleaning
        if cleaned and cleaned != Bicorpus.__BADToken():
            if lang == Lang.SOURCE: self.sourceWordCounts[cleaned] += 1
            else:                   self.destWordCounts[cleaned] += 1
        return cleaned

    def __processStrLine(self, line, lang):
        tokenCount = 0
        stripped = line.strip()
        cleaned = []
        
        for token in stripped.split(" "):
            cleanedToken = self.__processToken(token, lang) #FIXME: Don't do lowercasing of the text 

            if cleanedToken == Bicorpus.__BADToken(): return Bicorpus.__BADToken()
            if cleanedToken: cleaned.append(cleanedToken)

            tokenCount += 1

        tokenCount += 2 #For start and end sequences
        if lang == Lang.SOURCE: self.sourceTokensCount += tokenCount
        else:                   self.destTokensCount += tokenCount

        return " ".join( [Bicorpus.START()] + cleaned +  [Bicorpus.END()] )

    """
    def __processListLine(self, line, lang):
        print("Called processListLine()")
        tokenCount = 0
        for word in line:
            for character in word:
                self.__processToken(token, lang)
                tokenCount += 1

        tokenCount += 2
        if lang == Lang.SOURCE: self.sourceTokensCount += tokenCount
        else:                   self.destTokensCount += tokenCount

        return [Bicorpus.START()] + line + [Bicorpus.END()]
    """
        
    def __processSequence(self, line, lang):
        if type(line) == list: processed = self.__processListLine(line, lang)
        else:                  processed = self.__processStrLine(line, lang)
        return processed

    def __processBisequence(self, readIndex):
        """
        Return True if the sequences could be written and False otherwise.
	"""
        sourceLine = self.rawSource[readIndex]
        destLine = self.rawDest[readIndex]

        sourceProcessed = self.__processSequence(sourceLine, Lang.SOURCE)
        destProcessed = self.__processSequence(destLine, Lang.DEST)

        if sourceProcessed != Bicorpus.__BADToken() and destProcessed != Bicorpus.__BADToken():
            self.sourceLines.append( self.__processSequence(sourceLine, Lang.SOURCE) )
            self.destLines.append( self.__processSequence(destLine, Lang.DEST) )
            return True
        return False

    #Returns a tuple of (word to index dictionary, index to word dictionary)
    def __indexDict(self, lang, size):
       wordCounts = self.sourceWordCounts if lang == Lang.SOURCE else self.destWordCounts

       #print("lang=", lang, ", wordCounts=", wordCounts, sep = "")

       #Words in descending order of frequency
       words = sorted(wordCounts, key = wordCounts.get, reverse = True)

       wordMap = one2one(str, int, unknown_x = Bicorpus.UNK(), unknown_y = size)

       i = 0
       while i < size:
           wordMap[ words[i] ] = i
           i += 1

       Bicorpus.__addAnnotativeTokens(wordMap, size + 1) #Index i = size is already taken by unknown_y
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

        #print("numWords =", numWords)

        sourceMap = self.__indexDict(Lang.SOURCE, numWords)
        destMap = self.__indexDict(Lang.DEST, numWords)

        #print("sourceMap:", sourceMap)
        #print("destMap:", destMap)

        #print("len(sourceMap) =", len(sourceMap))
        #print("len(destMap) =", len(destMap))

        return sourceMap, destMap

    def __lazyLoad(self):
        if not(self.isLoaded()): 
            self.sourceMap, self.destMap = self.__genIndexDicts(self.maxWords, self.maxSequences)
############################Public functions##############################

    def isLoaded(self):
        return self.sourceMap != None

    def getTrainingLines(self):
        self.__lazyLoad()
        return self.sourceLines, self.destLines
    
    def getMaps(self):
        """
        Return the vocabulary mappings, generating them if need be.
        """
        self.__lazyLoad()
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
	Write sourceLines and destLines to a CTF file at an absolute path.
	"""
        self.__lazyLoad()
        writer = CTFFile(path, sourceLang, destLang, self.sourceMap, self.destMap)
        writer.writeSequences(self.sourceLines, self.destLines)
        writer.close()
        self.ctfPath = path

    def writeMapping(self, path, lang):
        """
	Write one of the mappings (determined by lang) to an absolute path.
	"""
        self.__lazyLoad()
        wordMap = self.sourceMap if lang == Lang.SOURCE else self.destMap
        wordMap.write(path)
        if lang == Lang.SOURCE: self.sourceMapPath = path
        else:                   self.destMapPath   = path

    def writeProps(self, path):
        """
        Write various properties associated with the text (# sequences, #tokens, etc.)
        """
        self.__lazyLoad()
        with open(path, "w") as propsFile:
            propsFile.write( str( len(self.sourceLines) ) + "\n")
            num_tokens = str( max(self.sourceTokensCount, self.destTokensCount) )
            propsFile.write( num_tokens + "\n" ) 

