import cntk as C
import numpy as np
import datetime
from scripts.bicorpus import Bicorpus

C.device.try_set_default_device(C.device.gpu(0))
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

numWords = 10000
#sourceVocabSize = vocabSize
#destVocabSize = vocabSize

#numSequences = 10000
training_ratio = 3 / 4
max_epochs = 1
#epoch_size = 50 #int(numSequences * training_ratio)


files = {}

sourceTraining = "corpora/europarl-v7.es-en.es"
destTraining = "corpora/europarl-v7.es-en.en"



with open(sourceTraining, "r", encoding = "utf-8") as sourceFile:
    sourceLines = sourceFile.readlines()

with open(destTraining, "r", encoding = "utf-8") as destFile:
    destLines = destFile.readlines()

print("Read the source data.", flush = True)

trainingCorp = Bicorpus(sourceLines, destLines, vocabSize = numWords) #,  numSequences = 10)

print("Done cleaning source data.", flush = True)

# In[3]:
del sourceLines
del destLines

cleanedSource, cleanedDest = trainingCorp.training_lines()

numSequences = len(cleanedSource)
epoch_size = int(numSequences * training_ratio)



sourceW2I, destW2I = trainingCorp.getW2IDicts()
sourceI2W, destI2W = trainingCorp.getI2WDicts()

#The "+ 1"s are for the UNK token; I will have to add that to Bicorpus later on
sourceVocabSize, destVocabSize = len(sourceW2I) + 1, len(destW2I) + 1

seq_start_index = destW2I[Bicorpus.start_token()]
seq_end_index = destW2I[Bicorpus.end_token()]
#seq_start = C.constant(np.asarray([i == seq_start_index for i in range(len(destW2I))], dtype = my_dtype))
seq_start = C.constant(np.asarray([i == seq_start_index for i in range(destVocabSize)], dtype = my_dtype))


print("sourceVocabSize =", sourceVocabSize)
print("destVocabSize =", destVocabSize)


# In[4]:

# Source and target inputs to the model
sourceAxis = C.Axis("sourceAxis")
destAxis = C.Axis("destAxis")
sourceSequence = C.layers.SequenceOver[sourceAxis]
destSequence = C.layers.SequenceOver[destAxis]


# In[5]:

#Returns a general sequence-to-sequence model
def create_model():
    embed = C.layers.Embedding(embedding_dim, name = "embed") if use_embedding else identity #Where is "identity defined?
    
    with C.layers.default_options(enable_self_stabilization = True, go_backwards = not use_attention):
        LastRecurrence = C.layers.Fold if not use_attention else C.layers.Recurrence
        encode = C.layers.Sequential([
            embed,
            C.layers.Stabilizer(),
            C.layers.For(range(num_layers - 1), lambda: C.layers.Recurrence(C.layers.GRU(hidden_dim))),
            LastRecurrence(C.layers.GRU(hidden_dim), return_full_state = True),
            C.layers.Label("encoded_h")                                  
        ])
    
    with C.layers.default_options(enable_self_stabilization = True):
        stab_in = C.layers.Stabilizer()
        rec_blocks = [C.layers.GRU(hidden_dim) for i in range(num_layers)]
        stab_out = C.layers.Stabilizer()
        proj_out = C.layers.Dense(destVocabSize, name = "out_proj")
        if use_attention:
            attention_model = C.layers.AttentionModel(attention_dim, name = "attention_model")
            
        @C.Function
        def decode(history, input):
            encoded_input = encode(input)
            r = history
            r = embed(r)
            r = stab_in(r)
            for i in range(num_layers):
                rec_block = rec_blocks[i]
                if i == 0:
                    if use_attention:
                        @C.Function
                        def gru_with_attention(dh, x):
                            h_att = attention_model(encoded_input.outputs[0], dh)
                            x = C.splice(x, h_att)
                            toReturn = rec_block(dh, x)
                            return toReturn
                        r = C.layers.Recurrence(gru_with_attention)(r)
                        
                    else:
                        r = C.layers.Recurrence(rec_block)(r)
                else:
                    r = C.layers.RecurrenceFrom(rec_block)( *(encoded_input.outputs + (r,)) )
            r = stab_out(r)
            r = proj_out(r)
            r = C.layers.Label("out_proj_out")(r)
            return r
                
        return decode


# In[6]:

def create_model_train(s2smodel):
    @C.Function
    def model_train(input, labels):
        past_labels = C.layers.Delay(initial_state = seq_start)(labels)
        return s2smodel(past_labels, input)
    return model_train


# In[7]:

#Model used in testing
def create_model_greedy(s2smodel):
    @C.Function
    @C.layers.Signature(sourceSequence[C.layers.Tensor[sourceVocabSize]])
    def model_greedy(input):
        unfold = C.layers.UnfoldFrom(lambda history: s2smodel(history, input) >> C.hardmax,
                                    until_predicate = lambda w: w[..., seq_end_index],
                                    length_increase = length_increase)
        return unfold(initial_state = seq_start, dynamic_axes_like = input)
    return model_greedy


# In[8]:

def create_criterion_function(model):
    @C.Function
    @C.layers.Signature(input=sourceSequence[C.layers.Tensor[sourceVocabSize]],
                        labels=destSequence[C.layers.Tensor[destVocabSize]]) #Should also be "sourceVocabSize?"
    def criterion(input, labels):
        # criterion function must drop the <s> from the labels
        postprocessed_labels = C.sequence.slice(labels, 1, 0) # <s> A B C </s> --> A B C </s>
        #print("input =", input)
        #print("postprocessed_labels =", postprocessed_labels)
        z = model(input, postprocessed_labels)
        #print("z =", z)
        ce = C.cross_entropy_with_softmax(z, postprocessed_labels) #labels)
        errs = C.classification_error(z, postprocessed_labels) #labels)
        return (ce, errs)

    return criterion


# In[9]:

def format_sequences(sequences, i2w):
    return [" ".join([i2w[np.argmax(w)] for w in s]) for s in sequences]

def debug_attention(model, input):
    q = C.combine([model, model.attention_model.attention_weights])
    #words, p = q(input) # Python 3
    words_p = q(input)
    words = words_p[0]
    p     = words_p[1]
    output_seq_len = words[0].shape[0]
    p_sq = np.squeeze(p[0][:output_seq_len,:,:]) # (batch, output_len, input_len, 1)
    opts = np.get_printoptions()
    np.set_printoptions(precision=5)
    print(p_sq)
    np.set_printoptions(**opts)


# In[10]:

def wordToOneHot(word, w2i):
    #The "plus 1" is for the <UNK> token
    vector = np.zeros(len(w2i) + 1, dtype = my_dtype)
    try:
        vector[w2i[word]] = 1
    except KeyError:
        vector[len(w2i)] = 1 #The one-hot vector for the <UNK> token
    return vector

def wordsToIndices(wordList, w2i):
    if type(wordList) == str: wordList = wordList.split(" ")
    indices = []
    for word in wordList:
        try:
            indices.append(w2i[word])
        except KeyError:
            indices.append(len(w2i)) #The one-hot vector for the <UNK> token
    return indices

def wordsToOneHot(wordList, w2i):
    if type(wordList) == str: wordList = wordList.split(" ")
    return np.stack([wordToOneHot(word, w2i) for word in wordList])

def sequencesToOneHot(sequences, w2i):
    return [wordsToOneHot(sequence, w2i) for sequence in sequences]


# In[ ]:

#The original had "vocab" as a parameter but never used it
#The original had CTF readers train_reader and validation_reader
def train(sourceW2I, destW2I, s2smodel, max_epochs, epoch_size):

    oneHotSource = sequencesToOneHot(cleanedSource, sourceW2I)
    oneHotDest = sequenceToOneHot(cleanedDest, destW2I)

    data = C.io.MinibatchSourceFromData(dict(x = oneHotSource, y = oneHotDest))


    model_train = create_model_train(s2smodel)
    criterion = create_criterion_function(model_train)
    #model_greedy = create_model_greedy(s2smodel)
    
    minibatch_size = 50
    lr = 0.001 if use_attention else 0.005


    learner = C.fsadagrad(model_train.parameters,
                         lr = C.learning_rate_schedule([lr]*2 + [lr/2]*3 +[lr/4], C.UnitType.sample, epoch_size),
                         momentum = C.momentum_as_time_constant_schedule(1100),
                         gradient_clipping_threshold_per_sample = 2.3,
                         gradient_clipping_with_truncation = True)


    trainer = C.Trainer(model.parameters, criterion, learner)                      #

    training_session(trainer = trainer,
                     mb_source = data,
                     model_inputs_to_streams = {model_train[""] : train_source
                     mb_size = minibatch_size, 
                     
    
"""
    total_samples = 0
    mbs = 0
    eval_freq = 100
    C.logging.log_number_of_parameters(model_train) ; print()
    progress_printer = C.logging.ProgressPrinter(freq = 30, tag = "Training")


    
    
    #sparse_to_dense = create_sparse_to_dense(sourceVocabSize)
    
    for epoch in range(max_epochs):
        mb_num = 0
        while total_samples < (epoch + 1) * epoch_size:
            print("Trained {0} of {1} samples at {2}".format(total_samples, epoch_size, datetime.datetime.now().strftime("%H:%M:%S")), flush = True)
            #print("total_samples =", total_samples)
            #print("mbs =", mbs)
            startIndex = mbs * minibatch_size
            endIndex = startIndex + minibatch_size
            
            #mb_train = train_reader.next_minibatch(minibatch_size)
            #trainer.train_minibatch({criterion.arguments[0]: mb_train[train_reader.streams.features],
                                     #criterion.arguments[1]: mb_train[train_reader.streams.labels]})
            
            sourceBatch = sequencesToOneHot(cleanedSource[startIndex:endIndex], sourceW2I)
            destBatch = sequencesToOneHot(cleanedDest[startIndex:endIndex], destW2I)
            trainer.train_minibatch({criterion.arguments[0]: sourceBatch,
                                     criterion.arguments[1]: destBatch
                                    })
            
            
            
            progress_printer.update_with_trainer(trainer, with_metric = True)
            
            #if mbs % eval_freq == 0:
                #mb_valid = valid_reader.next_minibatch(1)
                #e = model_greedy(mb_valid[valid_reader.streams.features])
                #
                ##Need to i2w to my own dictionary
                ##Really, I just need to add a function to my Bicorpus class
                #print(format_sequence(sparse_to_dense(mb_valid[valid_reader.streams.features]), sourceI2W))
                #print("-->")
                #print(format_sequences(e, destI2W))
                #
                #if use_attention:
                    #debug_attention(model_greedy, mb_valid[valid_reader.streams.features])
                    
            total_samples += minibatch_size #mb.train[train_reader.streams.labels].num_samples
            mbs += 1
                
    progress_printer.epoch_summary(with_metric = True)
"""

# In[ ]:


print("About to train model", flush = True)
model = create_model()
train(sourceI2W, destI2W, model, max_epochs, epoch_size)
print("Finished training", flush = True)

timeSuffix = datetime.datetime.now().strftime("%b_%d")
modelPath = "model_" + timeSuffix
model.save(modelPath)
print("Saved model to", modelPath, flush = True)


# In[ ]:

import inspect

def translate(text, model, sourceW2I, destI2W):
    #vectors = wordsToOneHot(text, sourceW2I)
    
    
    indices = wordsToIndices(text, sourceW2I)
    indices.insert(0, sourceW2I[Bicorpus.start_token()])
    indices.append(sourceW2I[Bicorpus.end_token()])
    
    query = C.Value.one_hot([indices], len(sourceW2I) + 1)
    
    print(inspect.signature(model))
    print(model.arguments)
    
    pred = model(query)
    pred = pred[0] # first sequence (we only have one) -> [len, vocab size]
    if use_attention:
        pred = np.squeeze(pred) # attention has extra dimensions

    # print out translation and stop at the sequence-end tag
    prediction = np.argmax(pred, axis=-1)
    #translation = [destI2W[i] for i in prediction]
    
    #print(translation)
    for i in prediction:
        if i == len(destI2W): print("UNK", flush = True)
        else: print(destI2W[i], flush = True)
    
    

def debugging(trained_model):
    model = create_model_greedy(trained_model)
    
    query = input("Enter a Spanish phrase (or just <Enter> to quit): ").strip()
    while query:
        translate(query, model, sourceW2I, destI2W)
        query = input("Enter a Spanish phrase (or just <Enter> to quit): ").strip()
        
print("Trying simple query on model", flush = True)
translate("ella es buena", model, sourceW2I, destI2W)

# In[ ]:

#debugging(model)

