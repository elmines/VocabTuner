#!/bin/bash

TC_MODEL=$1
CORPUS=$2
TC_CORPUS=$3

if [ ! -e $TC_CORPUS ]
then
  /home/ualelm/mosesdecoder/scripts/recaser/truecase.perl \
    --model $TC_MODEL \
    < $CORPUS \
    > $TC_CORPUS
fi
