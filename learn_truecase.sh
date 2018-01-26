#!/bin/bash

CORPUS=$1
MODEL_PATH=$2

if [ ! -e $OUTPUT ]
then
  #echo "About to train truecase model for $LANG_ABBR"
  /home/ualelm/mosesdecoder/scripts/recaser/train-truecaser.perl \
    --model $MODEL_PATH \
    --corpus $CORPUS
fi

