#!/bin/bash

RAW_SOURCE=$1
RAW_DEST=$2

#Outputs
CODES=$3
SOURCE_VOCAB=$4
DEST_VOCAB=$5

#A constant (for now)
NUM_SEQUENCES=60000

if [ ! -e $CODES ]
then
    python3 ~/subword-nmt/learn_joint_bpe_and_vocab.py \
    --input $RAW_SOURCE $RAW_DEST \
    -s $NUM_SEQUENCES \
    -o $CODES \
     --write-vocabulary $SOURCE_VOCAB $DEST_VOCAB
fi

