import cntk as C
import numpy as np
import os
import sys
#Local modules
modulesPath = "scripts"
modulesPath = os.path.abspath(os.path.join(modulesPath))
if modulesPath not in sys.path: sys.path.append(modulesPath)
from bicorpus import Bicorpus
from one2one import one2one
import europarse

my_dtype = np.float32
length_increase = 1.5

constPath = "Nov_15_12_41.cmf"
sourceLang = "es"
destLang = "en"

paths = europarse.getPaths(sourceLang, destLang)
#sourceMapPath, destMapPath, ctfPath = europarse.getPaths(sourceLang, destLang)
sourceMapPath = paths.getSourceMapPath()
destMapPath = paths.getDestMapPath()
ctfPath = paths.getCtfPath()

sourceMapping = one2one.load(sourceMapPath)
destMapping = one2one.load(destMapPath)
sourceVocabSize = len(sourceMapping)
destVocabSize = len(destMapping)

seqStartIndex = destMapping[Bicorpus.START()]
seqEndIndex = destMapping[Bicorpus.END()]
seqStart = C.Constant( np.asarray( [i == seqStartIndex for i in range(destVocabSize) ] , dtype = my_dtype) )

sourceAxis = C.Axis("sourceAxis")
destAxis = C.Axis("destAxis")
sourceSequence = C.layers.SequenceOver[sourceAxis]
destSequence = C.layers.SequenceOver[destAxis]

modelPath = sys.argv[1] if len(sys.argv) > 1 else constPath

def create_model_greedy(s2smodel):
    # model used in (greedy) decoding (history is decoder's own output)
    @C.Function
    @C.layers.Signature(sourceSequence[C.layers.Tensor[sourceVocabSize]])
    def model_greedy(input): # (input*) --> (word_sequence*)

        # Decoding is an unfold() operation starting from sentence_start.
        # We must transform s2smodel (history*, input* -> word_logp*) into a generator (history* -> output*)
        # which holds 'input' in its closure.
        unfold = C.layers.UnfoldFrom(lambda history: s2smodel(history, input) >> C.hardmax,
                            # stop once sentence_end_index was max-scoring output
                            until_predicate=lambda w: w[...,seqEndIndex],
                            length_increase=length_increase)

        return unfold(initial_state=seqStart, dynamic_axes_like=input)
    return model_greedy

s2smodel = C.Function.load(modelPath)
greedy_model = create_model_greedy(s2smodel)

def printPhrases(sourcePhrase, destPhrase):
    for i in range(1, len(sourcePhrase) - 1):
        print(sourcePhrase[i], end = " ")

    print(" -->", destPhrase)

    #for i in range(len(destPhrase) - 1):
        #print(" ", destPhrase[i], sep = "", end = "")

    print()

def debugging(greedy_model):
    sourcePhrases = [ ["<s>", "yo", "soy", "presidenta", "</s>"], ["<s>", "ella", "ama", "europa", "</s>"], ["<s>", "nosotros", "necesitamos", "ayuda", "</s>"] ] 


    for sourcePhrase in sourcePhrases:
        indices = [sourceMapping[word] for word in sourcePhrase]
        query = C.Value.one_hot(indices, sourceVocabSize)

        pred = greedy_model(query)
        pred = np.squeeze(pred[0])

        words = [ destMapping[np.argmax(one_hot)] for one_hot in pred ]
        phrase = " ".join(words)
        printPhrases(sourcePhrase, phrase)


debugging(greedy_model)
