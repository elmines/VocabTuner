import cntk as C
import numpy as np
import datetime
from scripts.bicorpus import Bicorpus
import scripts.bicorpus


devices = c.cntk_py.all_devices()
for device in devices:
    if device.type():
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


def dataPreprocessed(sourceLang, destLang):
   europarse.parseCorpora(soruceLang, destLang, maxWords = maxWords, maxSequences = maxSequences)


files = {}

sourceTraining = "corpora/europarl-v7.es-en.es"
destTraining = "corpora/europarl-v7.es-en.en"




trainingCorp = Bicorpus(sourceLines, destLines, maxWords = maxWords, maxSequences = maxSequences)
sourceLines, destLines = trainingCorp.training_lines()

print(sourceLines[:5])
print(destLines[:5])

#Epoch Size
numSequences = len(cleanedSource)
epoch_size = int(numSequences * training_ratio)

