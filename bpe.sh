#!/bin/bash

DATA_DIR=$1
SOURCE_CORP=$2
DEST_CORP=$3
SUFFIX=$4

#Constants (for now)
NUM_SEQUENCES=5000
THRESH=50

SOURCE_LANG=`basename -s $SUFFIX $SOURCE_CORP`
DEST_LANG=`basename -s $SUFFIX $DEST_CORP`

CODES=$DATA_DIR/${SOURCE_LANG}-${DEST_LANG}.codes
SOURCE_VOCAB=$DATA_DIR/${SOURCE_LANG}.vocab
DEST_VOCAB=$DATA_DIR/${DEST_LANG}.vocab

if [ ! -e $CODES ]
then
    python3 ~/subword-nmt/learn_joint_bpe_and_vocab.py \
    --input $SOURCE_CORP $DEST_CORP \
    -s $NUM_SEQUENCES \
    -o $CODES \
     --write-vocabulary $SOURCE_VOCAB $DEST_VOCAB
fi

BPE_SOURCE=${SOURCE_CORP}.bpe
BPE_DEST=${DEST_CORP}.bpe

if [ ! -e $BPE_SOURCE ]
then
    python3 ~/subword-nmt/apply_bpe.py \
    --codes $CODES \
    --vocabulary $SOURCE_VOCAB \
    --vocabulary-threshold $THRESH \
    < $SOURCE_CORP \
    > $BPE_SOURCE

    python3 ~/subword-nmt/apply_bpe.py \
    --codes $CODES \
    --vocabulary $DEST_VOCAB \
    --vocabulary-threshold $THRESH \
    < $DEST_CORP \
    > $BPE_DEST
fi

echo $BPE_SOURCE $BPE_DEST $SOURCE_VOCAB $DEST_VOCAB
