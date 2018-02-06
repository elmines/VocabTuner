#!/bin/bash

module load boost/1.58.0 cuda/8.0.44

SOURCE_LANG=es
DEST_LANG=en

EXPERIMENT=~/VocabTuner/experiments/${SOURCE_LANG}-${DEST_LANG}

MODEL=$EXPERIMENT/models/es-en.npz
JOINT_VOCAB=$EXPERIMENT/data/es-en.yml


SOURCE=$EXPERIMENT/"eval"/${SOURCE_LANG}.test.tok.tc.bpe
TRANSLATION=$EXPERIMENT/"eval"/${DEST_LANG}.trans

if [ ! -e $TRANSLATION ]
then
  ~/marian/build/marian-decoder \
    --models $MODEL \
    --vocabs $JOINT_VOCAB $JOINT_VOCAB \
    --input $SOURCE \
    --devices 0 \
    > $TRANSLATION
fi


