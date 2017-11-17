import cntk as C
import numpy as np

import os
import sys
import datetime


#Local modules
modulesPath = "scripts"
modulesPath = os.path.abspath(os.path.join(modulesPath))
if modulesPath not in sys.path: sys.path.append(modulesPath)
from one2one import one2one
import europarse
from bicorpus import Bicorpus
from corp_paths import CorpPaths

def printCom(string):
    print( str(C.distributed.Communicator.rank()), "> ",  string, sep = "" )

devices = C.device.all_devices()

gpuAvailable = False
for device in devices:
    if device.type(): #Means a GPU is available
        gpuAvailable = True
        break

if gpuAvailable:
    C.device.try_set_default_device(C.device.gpu( C.distributed.Communicator.rank() ) )

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
training_ratio = 1 #3 / 4
minibatch_size = 1
max_epochs = 1


#Language hyperparameters
sourceLang = "es"
destLang = "en"

if len(sys.argv) > 1:
    if len(sys.argv) == 2:
        langs = sys.argv[1].split("-")
        sourceLang, destLang = langs[0], langs[1]
    else:
        sourceLang, destlang = sys.argv[1], sys.argv[2]

def create_reader(ctfPath, sourceLang, destLang, sourceVocabSize, destVocabSize):
    sourceStream = C.io.StreamDef(field = sourceLang, shape = sourceVocabSize, is_sparse = True)
    destStream = C.io.StreamDef(field = destLang, shape = destVocabSize, is_sparse = True)

    deserializer = C.io.CTFDeserializer(ctfPath, C.io.StreamDefs(labels = destStream, features = sourceStream))
    reader = C.io.MinibatchSource(deserializer, randomize = 0, max_sweeps = 1)
    return reader

paths = europarse.getPaths(sourceLang, destLang)

sourceMapPath = paths.getSourceMapPath()
destMapPath = paths.getDestMapPath()
ctfPath = paths.getCtfPath()

sourceMapping = one2one.load(paths.getSourceMapPath())
destMapping = one2one.load(paths.getDestMapPath())
sourceVocabSize = len(sourceMapping)
destVocabSize = len(destMapping)
trainingReader = create_reader(ctfPath, sourceLang, destLang, sourceVocabSize, destVocabSize)


#########################MODEL VARIABLES##########################
sourceAxis = C.Axis("sourceAxis")
destAxis = C.Axis("destAxis")
sourceSequence = C.layers.SequenceOver[sourceAxis]
destSequence = C.layers.SequenceOver[destAxis]
#Sequence start and end marks (only for the decoder?)
seqStartIndex = destMapping[Bicorpus.START()]
seqEndIndex = destMapping[Bicorpus.END()]
seqStart = C.Constant( np.asarray( [i == seqStartIndex for i in range(destVocabSize) ] , dtype = my_dtype) )


# create the s2s model
def create_model(): # :: (history*, input*) -> logP(w)*

    # Embedding: (input*) --> embedded_input*
    embed = C.layers.Embedding(embedding_dim, name='embed') if use_embedding else identity

    # Encoder: (input*) --> (h0, c0)
    # Create multiple layers of LSTMs by passing the output of the i-th layer
    # to the (i+1)th layer as its input
    # Note: We go_backwards for the plain model, but forward for the attention model.
    with C.layers.default_options(enable_self_stabilization=True, go_backwards=not use_attention):
        LastRecurrence = C.layers.Fold if not use_attention else C.layers.Recurrence
        encode = C.layers.Sequential([
            embed,
            C.layers.Stabilizer(),
            C.layers.For(range(num_layers-1), lambda:
                C.layers.Recurrence(C.layers.LSTM(hidden_dim))),
            LastRecurrence(C.layers.LSTM(hidden_dim), return_full_state=True),
            (C.layers.Label('encoded_h'), C.layers.Label('encoded_c')),
        ])

    # Decoder: (history*, input*) --> unnormalized_word_logp*
    # where history is one of these, delayed by 1 step and <s> prepended:
    #  - training: labels
    #  - testing:  its own output hardmax(z) (greedy decoder)
    with C.layers.default_options(enable_self_stabilization=True):
        # sub-layers
        stab_in = C.layers.Stabilizer()
        rec_blocks = [C.layers.LSTM(hidden_dim) for i in range(num_layers)]
        stab_out = C.layers.Stabilizer()
        proj_out = C.layers.Dense(destVocabSize, name='out_proj')
        # attention model
        if use_attention: # maps a decoder hidden state and all the encoder states into an augmented state
            attention_model = C.layers.AttentionModel(attention_dim,
                                                      name='attention_model') # :: (h_enc*, h_dec) -> (h_dec augmented)
        # layer function
        @C.Function
        def decode(history, input):
            encoded_input = encode(input)
            r = history
            r = embed(r)
            r = stab_in(r)
            for i in range(num_layers):
                rec_block = rec_blocks[i]   # LSTM(hidden_dim)  # :: (dh, dc, x) -> (h, c)
                if use_attention:
                    if i == 0:
                        @C.Function
                        def lstm_with_attention(dh, dc, x):
                            h_att = attention_model(encoded_input.outputs[0], dh)
                            x = C.splice(x, h_att)
                            return rec_block(dh, dc, x)
                        r = C.layers.Recurrence(lstm_with_attention)(r)
                    else:
                        r = C.layers.Recurrence(rec_block)(r)
                else:
                    # unlike Recurrence(), the RecurrenceFrom() layer takes the initial hidden state as a data input
                    r = C.layers.RecurrenceFrom(rec_block)(*(encoded_input.outputs + (r,))) # :: h0, c0, r -> h
            r = stab_out(r)
            r = proj_out(r)
            r = C.layers.Label('out_proj_out')(r)
            return r

    return decode


#######################Training wrapper for Model###########################
def create_model_train(s2smodel):
    # model used in training (history is known from labels)
    # note: the labels must NOT contain the initial <s>
    @C.Function
    def model_train(input, labels): # (input*, labels*) --> (word_logp*)

        # The input to the decoder always starts with the special label sequence start token.
        # Then, use the previous value of the label sequence (for training) or the output (for execution).
        past_labels = C.layers.Delay(initial_state=seqStart)(labels)
        return s2smodel(past_labels, input)
    return model_train


#######################Testing wrapper for model##############################
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


def create_criterion_function(model):
    @C.Function
    @C.layers.Signature(input=sourceSequence[C.layers.Tensor[sourceVocabSize]],
                        labels=destSequence[C.layers.Tensor[destVocabSize]])
    def criterion(input, labels):
        # criterion function must drop the <s> from the labels
        postprocessed_labels = C.sequence.slice(labels, 1, 0) # <s> A B C </s> --> A B C </s>
        z = model(input, postprocessed_labels)
        ce = C.cross_entropy_with_softmax(z, postprocessed_labels)
        errs = C.classification_error(z, postprocessed_labels)
        return (ce, errs)

    return criterion


# Given a vocab and tensor, print the output
def format_sequences(sequences, i2w):
    return [" ".join([i2w[np.argmax(w)] for w in s]) for s in sequences]

# to help debug the attention window
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


#def train(train_reader """, valid_reader, vocab, i2w""",s2smodel, max_epochs, epoch_size):
def train(train_reader, s2smodel, max_epochs, epoch_size):

    # create the training wrapper for the s2smodel, as well as the criterion function
    model_train = create_model_train(s2smodel)
    criterion = create_criterion_function(model_train)

    # also wire in a greedy decoder so that we can properly log progress on a validation example
    # This is not used for the actual training process.
    model_greedy = create_model_greedy(s2smodel)


    #mb_source = C.io.MinibatchSource(train_reader, randomize = True) # multithreaded_deserializer = True)

    # Instantiate the trainer object to drive the model training
    #minibatch_size = 72
    lr = 0.001 if use_attention else 0.005

    lr_list = [lr] * 2 + [lr/2] * 3 + [lr/4]
    lr_schedule = C.learning_parameter_schedule(lr_list, minibatch_size = minibatch_size, epoch_size = epoch_size)

    learner = C.fsadagrad(model_train.parameters,
                          lr = lr_schedule,
                          momentum = C.momentum_as_time_constant_schedule(1100),
                          gradient_clipping_threshold_per_sample=2.3,
                          gradient_clipping_with_truncation=True)



    parallelLearner = C.distributed.data_parallel_distributed_learner(
       learner = learner,
       num_quantization_bits = 32,
       distributed_after = 0
    )


    trainer = C.Trainer(None, criterion, parallelLearner)

    # Get minibatches of sequences to train with and perform model training
    total_samples = 0
    mbs = 0
    eval_freq = 100

    # print out some useful training information
    #C.logging.log_number_of_parameters(model_train) ; print()
    #progress_printer = C.logging.ProgressPrinter(freq=30, tag='Training')

    # a hack to allow us to print sparse vectors
    #sparse_to_dense = create_sparse_to_dense(input_vocab_dim)

    printCom("Instantiating training session.")

    C.training_session(
         trainer = trainer, mb_source = train_reader,
	 model_inputs_to_streams = {criterion.arguments[0]: train_reader.streams.features, criterion.arguments[1]: train_reader.streams.labels},
	 mb_size = minibatch_size,
	 progress_frequency = minibatch_size,
         cv_config = None,
	 checkpoint_config = None,
	 test_config = None
    ).train()


    timeSuffix = datetime.datetime.now().strftime("%b_%d_%H_%M")  # done: save the final model
    model_path = timeSuffix + ".cmf"
    printCom("Saving final model to '%s'" % model_path)
    s2smodel.save(model_path)
    printCom("%d epochs complete." % max_epochs)

def train_model(sourceMapping, destMapping, paths):
    with open(paths.getPropsPath(), "r") as props:
       numSequences = int( props.readline() )
       printCom(numSequences)

    model = create_model()
    epoch_size = numSequences * training_ratio
    train(trainingReader, model, max_epochs, epoch_size)

def debugging(s2smodel):
    model_greedy = create_model_greedy(s2smodel);


train_model(sourceMapping, destMapping, paths)
printCom("About to finalize communicator.")
C.distributed.Communicator.finalize()

