from one2one import one2one

class bpe_corpus:
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
    dicts = None
    charset = None

    def __init__(self, base_corpus):
        """
        base_corpus -- the original text
            Currently may only be a list of strs.
        """
           
        self.views = {}
        self.dicts = {}
        self.charset = set()
        self.__initial_pass(base_corpus)

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
        return "</s>"

 

    def __initial_pass(base_corpus):
        origView = {}
        origDict = one2one
        self.views[0] = origView
        self.dicts[0] = origDict

        for line in base_corpus:
            words = []
            for word in line:
               chars = word.split(" ")
               self.add_bigrams(

               for i range(len(token)): 
                   


