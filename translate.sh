#!/bin/bash

module load boost/1.58.0 cuda/8.0.44

SOURCE_LANG=es
DEST_LANG=en

EXPERIMENT=~/VocabTuner/experiments/es-en

MODEL=$EXPERIMENT/models/es-en.iter90000.npz
JOINT_VOCAB=$EXPERIMENT/data/es-en.yml


SOURCE=$EXPERIMENT/data/${SOURCE_LANG}.test
TRANSLATION=$EXPERIMENT/data/${DEST_LANG}.test.trans

if [ ! -e $TRANSLATION ]
then
  marian-decoder \
    --models $MODEL \
    --vocabs $JOINT_VOCAB $JOINT_VOCAB \
    --input $SOURCE \
    --devices 0 \
    > $TRANSLATION
fi


