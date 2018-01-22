#!/bin/bash

DATA_DIR=$1
CORPUS=$2
SUFFIX=$3

#Outputs truecased file path of truecased CORPUS
LANG_ABBR=`basename -s $SUFFIX $CORPUS`
TC_CORPUS=${CORPUS}.tc
TC_MODEL=truecase-model.$LANG_ABBR


if [ ! -e $TC_MODEL ]
then
  #echo "About to train truecase model for $LANG_ABBR"
  /home/ualelm/mosesdecoder/scripts/recaser/train-truecaser.perl \
    --model $DATA_DIR/$TC_MODEL \
    --corpus $CORPUS
fi

if [ ! -e $TC_CORPUS ]
then
  /home/ualelm/mosesdecoder/scripts/recaser/truecase.perl \
    --model $DATA_DIR/$TC_MODEL \
    < $CORPUS \
    > $TC_CORPUS
fi

echo $TC_CORPUS
