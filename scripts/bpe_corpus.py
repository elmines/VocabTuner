#from bicorpus import Bicorpus

class BPECorpus:
    """
    Class for maintaining multiple states, each given a certain number of BPE operations.

    views -- An int->list dictionary, where the integer is the number
        of BPE merges and the list is the text corpus.

        A given text corpus has as an element a list of strings, where
        each string is a \"word\" with its individual symbols delmited
        by spaces. For example, the word \"imagination\" with no BPE
        merges would be represented as \"i m a g i n a t i o n\".
        If a few BPE operations had been applied across the corpus
        resulting in the subword unit \"tion\", the word's represenation
        would be \"i m a g i n a tion\".

        Thus, any given token within the string is a subword unit.

    dicts -- An int->one2one dictionary, where the integer is the number
        of BPE merges and the one2one is a mapping between subword units
        and one-hot encoded vector indices for the corpus.

    charset -- A simple str list, maintaining a list of all the different
        characters in the corpus. When generating dicts[i] for any given
        i, one should add the elements of charset to dicts[i] at the end,
        so that no unknown token is every encountered in translation.
        

    """
    
    views = None
    sourceCharset = None
    destCharset = None

    def __init__(self, sourceLines, destLines):
        """
        base_corpus -- the original text
            Currently may only be a list of strs.
        """
           
        self.views = {}
        self.charset = set()
        self.__initial_pass(sourceLines, destLines)

    @staticmethod
    def unknown_token():
        """Token for representing unknown words in dicts"""
        return "<UNK>"

    @staticmethod
    def sequence_start():
        """Token for representing the start of a sequence for an RNN."""
        return "<s>"

    @staticmethod
    def sequence_end():
        """Token for representing the end of a sequence for an RNN."""
        return "</s>"

    @staticmethod
    def space():
        """Token for representing a space between words (not just subword units) for an RNN."""
        return "<SPACE>"

    @staticmethod
    def __whitespaceChars():
        return " \n\r\t"

 

    def __initial_pass(self, sourceLines, destLines):

        splitSource = BPECorpus.__splitCorpTokens(sourceLines)
        splitDest = BPECorpus.__splitCorpTokens(destLines)


        print(splitSource)
        print(splitDest)


    @staticmethod
    def __splitCorpTokens(lines):

        #splitted = [ BPECorpus.__splitLine(line) for line in lines ]
        splitted = []
        limit = 10
        for i in range(limit):
            splitted.append( BPECorpus.__splitLine(lines[i]) )

        return splitted

    @staticmethod
    def __splitLine(line):
        splitted = []

        insertSpace = False

        for token in line.strip().split(" "):
           if insertSpace: splitted.append( BPECorpus.space() )
           else:           insertSpace = True
           splitted.append( BPECorpus.__splitToken(token) )

        """
        #Trim trail
        if splitted[-1][-1] = '\n':
            splitted[-1] = splitted[:-1]
        """
        return splitted

    @staticmethod
    def __splitToken(token):
        splitted = []

        for c in token:
            splitted.append(c)

        return " ".join(splitted)

                   


