#!/bin/bash

# $1 = Data directory
# $2 = Source Language
# $3 = Destination Language

DATA_ROOT=$1
SOURCE_LANG=$2
DEST_LANG=$3

MOSES_SCRIPTS=/home/ualelm/mosesdecoder/scripts
BPE_SCRIPTS=/home/ualelm/subword-nmt

DATA_DIR=$DATA_ROOT/${SOURCE_LANG}-${DEST_LANG}
if [ ! -e $DATA_DIR ]
then
  DATA_DIR=$DATA_ROOT/${DEST_LANG}-${SOURCE_LANG}
  if [ ! -e $DATA_DIR ]
  then
    echo "Error: no data for language pair ${SOURCE_LANG}-${DEST_LANG}"
    exit 1
  fi
fi

RAW_SOURCE=$DATA_DIR/${SOURCE_LANG}.train
RAW_DEST=$DATA_DIR/${DEST_LANG}.train

#echo "Raw text files:"
#echo $RAW_SOURCE
#echo $RAW_DEST


TOK_SOURCE=`./tokenize.sh $RAW_SOURCE`

echo "Tokenized data:"
echo $TOK_SOURCE

TC_SOURCE=`./truecase.sh $DATA_DIR $TOK_SOURCE .train.tok`
echo "Truecased data:"
echo $TC_SOURCE

