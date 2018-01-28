#!/bin/bash

COMMAND=$1
SOURCE_LANG=$2
DEST_LANG=$3

#Commands
ALL="all"
PREPROCESS="preprocess"
TRAIN="train"



EXPERIMENT=`pwd`/experiments/$SOURCE_LANG-$DEST_LANG


if [ $COMMAND = $ALL ] || [ $COMMAND = $PREPROCESS ]
then
    ./preprocess.sh $EXPERIMENT/data $SOURCE_LANG $DEST_LANG
fi

if [ $COMMAND = $ALL ] || [ $COMMAND = $TRAIN ]
then
     ./train.sh $EXPERIMENT $SOURCE_LANG $DEST_LANG
fi
