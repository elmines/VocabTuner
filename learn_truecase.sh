#!/bin/bash

CORPUS=$1
TC_MODEL=$2

if [ ! -e $TC_MODEL ]
then
  #echo "About to train truecase model for $LANG_ABBR"
  /home/ualelm/mosesdecoder/scripts/recaser/train-truecaser.perl \
    --model $TC_MODEL \
    --corpus $CORPUS
fi

