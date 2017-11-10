import cntk as C
import numpy as np

import os
import sys
import datetime


#Local modules
modulesPath = "scripts"
modulesPath = os.path.abspath(os.path.join(modulesPath))
if modulesPath not in sys.path: sys.path.append(modulesPath)
from bicorpus import Bicorpus
from one2one import one2one
import europarse

devices = C.device.all_devices()
for device in devices:
    if device.type(): #Means a GPU is available
        C.device.try_set_default_device(C.device.gpu(0))
        print("Found GPU and set as default device.")
        print(device.get_gpu_properties(device))


C.cntk_py.set_fixed_random_seed(0)

#Model hyperparameters
my_dtype = np.float32
hidden_dim = 512
num_layers = 2
attention_dim = 128
use_attention = True
use_embedding = True
embedding_dim = 200
length_increase = 1.5

#Data hyperparameters
maxWords = 10000
maxSequences = 10000
training_ratio = 1 #3 / 4
minibatch_size = 32
max_epochs = 1


#Language hyperparameters
sourceLang = "es"
destLang = "en"


trainingCorp, sourceMapPath, destMapPath, ctfPath = europarse.parseCorpora(sourceLang, destLang, maxWords = maxWords, maxSequences = maxSequences)


cleanedSource, cleanedDest = trainingCorp.training_lines()


#Epoch Size
numSequences = len(cleanedSource)
epoch_size = int(numSequences * training_ratio)
print(epoch_size)

sourceMapping = one2one.load(sourceMapPath)
destMapping = one2one.load(destMapPath)
sourceVocabSize = len(sourceMapping)
destVocabSize = len(destMapping)
print(ctfPath)


sourceVector = C.input_variable(sourceVocabSize)
destVector = C.input_variable(destVocabSize)

def create_reader(ctfPath, sourceLang, destLang, sourceVocabSize, destVocabSize):
    sourceStream = C.io.StreamDef(field = sourceLang, shape = sourceVocabSize, is_sparse = True)
    destStream = C.io.StreamDef(field = destLang, shape = destVocabSize, is_sparse = True)

    deserializer = C.io.CTFDeserializer(ctfPath, C.io.StreamDefs(labels = destStream, features = sourceStream))

    reader = C.io.MinibatchSource(deserializer, randomize = 0, max_sweeps = 1)


    mb = reader.next_minibatch(4)
    
    print( mb )
    
    featureData = mb[reader.streams.features]
    labelData = mb[reader.streams.labels]
    
    print( featureData )
    print( labelData )

    featureSequences = featureData.as_sequences(sourceVector) 
    labelSequences = labelData.as_sequences(destVector)
    print(featureSequences) 
    print(labelSequences)

    print( type(featureSequences) )
    print( type(labelSequences) ) 

    print( featureSequences[0] )
    print( labelSequences[0] ) 

    return reader

reader = create_reader(ctfPath, sourceLang, destLang, sourceVocabSize, destVocabSize)


