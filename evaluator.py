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

constPath = "Nov_11_19_38.cmf"
sourceLang = "es"
destLang = "en"

sourceMapPath, destMapPath, ctfPath = europarse.getPaths(sourceLang, destLang)

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

def debugging(greedy_model):
    sourcePhrase = ["<s>", "yo", "soy", "presidenta", "</s>"]

    indices = [sourceMapping[word] for word in sourcePhrase]
    query = C.Value.one_hot(indices, sourceVocabSize)

    pred = greedy_model(query)
    pred = np.squeeze(pred[0])

    words = [ destMapping[np.argmax(one_hot)] for one_hot in pred ]
    phrase = " ".join(words)
    print(phrase)


debugging(greedy_model)
