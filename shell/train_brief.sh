#!/bin/bash

SOURCE=$1
DEST=$2
SOURCE_VOCAB=$3
DEST_VOCAB=$4
MODEL_PREFIX=$5
LOG=$6

SEED=1

let EPOCH_SIZE=2**18     #Approximately 250,000
let MINIBATCH_SIZE=2**5  #Hopefully it's evident that is 32
let NUM_MINIBATCHES=$EPOCH_SIZE/$MINIBATCH_SIZE
let DISP_FREQ=MINIBATCH_SIZE*8

~/marian/build/marian \
    --type s2s \
    --train-sets $SOURCE $DEST \
    --vocabs $SOURCE_VOCAB $DEST_VOCAB \
    --model $MODEL_PREFIX \
    --dim-emb 256 \
    --dim-rnn 512 \
    --tied-embeddings \
    --layer-normalization \
    --skip \
    --mini-batch $MINIBATCH_SIZE \
    --after-batches 5 \
    --workspace 8192 \
    --disp-freq $DISP_FREQ \
    --quiet --log $LOG \
    --seed $SEED \
    --device 0 

    #--after-epochs 1 \
