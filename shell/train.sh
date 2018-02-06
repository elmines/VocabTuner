#!/bin/bash

module load boost/1.58.0 cuda/8.0.44

EXPERIMENT=$1
SOURCE_LANG=$2
#Declare DEST_LANG as global because the validation scripts needs it but can't take parameters
export DEST_LANG=$3 

DATA_DIR=data
MODEL_DIR=models
LOG_DIR=logs

#Data
TRAIN_SOURCE=$EXPERIMENT/$DATA_DIR/${SOURCE_LANG}.train.tok.tc.bpe
TRAIN_DEST=$EXPERIMENT/$DATA_DIR/${DEST_LANG}.train.tok.tc.bpe
DEV_SOURCE=$EXPERIMENT/$DATA_DIR/${SOURCE_LANG}.dev.tok.tc.bpe
DEV_DEST=$EXPERIMENT/$DATA_DIR/${DEST_LANG}.dev.tok.tc.bpe
JOINT_VOCAB=$EXPERIMENT/$DATA_DIR/${SOURCE_LANG}-${DEST_LANG}.yml

#Models
MODELS=$EXPERIMENT/$MODEL_DIR/${SOURCE_LANG}-${DEST_LANG}.npz

#Training Configuration
CONFIG=$EXPERIMENT/${SOURCE_LANG}-${DEST_LANG}.config.yml

#Logging
TRAIN_LOG=$EXPERIMENT/$LOG_DIR/${SOURCE_LANG}-${DEST_LANG}.train.log
DEV_LOG=$EXPERIMENT/$LOG_DIR/${SOURCE_LANG}-${DEST_LANG}.dev.log

echo "MODELS=" $MODELS
echo "CONFIG=" $CONFIG
echo "TRAIN_LOG=" $TRAIN_LOG
echo "DEV_LOG=" $DEV_LOG

if [ ! -e $CONFIG ]
then
    #Note the lack of --dim-vocabs. We just let JOINT_VOCAB determine the vocabulary sizes

    ~/marian/build/marian \
        --type s2s \
        --train-sets $TRAIN_SOURCE $TRAIN_DEST \
        --valid-sets $DEV_SOURCE   $DEV_DEST \
        --vocabs $JOINT_VOCAB $JOINT_VOCAB \
        --model  $MODELS \
        --enc-depth 4 --enc-type alternating --enc-cell lstm --enc-cell-depth 2 \
        --dec-depth 4                        --dec-cell lstm --dec-cell-base-depth 4 --dec-cell-high-depth 2 \
        --tied-embeddings \
        --layer-normalization \
        --skip \
        --mini-batch 64 \
        --workspace 8192 \
        --dropout-rnn 0.2 --dropout-src 0.1 --exponential-smoothing \
        --early-stopping 5 --disp-freq 1000 \
        --device 0 \
        --log $TRAIN_LOG --valid-log $DEV_LOG \
        --dump-config > $CONFIG
fi

~/marian/build/marian --config $CONFIG
