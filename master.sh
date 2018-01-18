#!/bin/bash

SOURCE_LANG=$1
DEST_LANG=$2

DATA_ROOT=data
DATA_DIR=$DATA_ROOT/$SOURCE_LANG-$DEST_LANG

if [ ! -e $DATA_DIR ]
then
  DATA_DIR=$DATA_ROOT/$DEST_LANG-$SOURCE_LANG

  if [ ! -e $DATA_DIR ]
  then
    echo "Error: no data for that language pair $SOURCE_LANG-$DEST_LANG"
    exit 1
  fi
fi

#Preprocess command
    #Tokenize input data
    #Truecase input data
    #Learn BPE segments from input data
    #Apply BPE to input data

#Train command
    #Set up Marian model
    #Train model

#Evaluation command
    #Preprocess test data
    #Translate test data
    #Undo BPE merges on test data
    #Undo Truecasing on test data
    #Undo tokenization on test data

    #Wrap test data in SGML
    #Evaluate test data against actual corpus 
