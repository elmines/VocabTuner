#!/bin/bash

# $1 = Data directory
# $2 = Source Language
# $3 = Destination Language

DATA_ROOT=$1
SOURCE_LANG=$2
DEST_LANG=$3

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

TOK_SOURCE=`./tokenize.sh $RAW_SOURCE`
TOK_DEST=`./tokenize.sh $RAW_DEST`
echo "Tokenized data:" $TOK_SOURCE $TOK_DEST

TC_SOURCE=`./truecase.sh $DATA_DIR $TOK_SOURCE .train.tok`
TC_DEST=`./truecase.sh $DATA_DIR $TOK_DEST .train.tok`
echo "Truecased data:" $TC_SOURCE $TC_DEST

BPE_FILES=(`./bpe.sh $DATA_DIR $TC_SOURCE $TC_DEST .train.tok.tc`)
BPE_SOURCE=${BPE_FILES[0]}
BPE_DEST=${BPE_FILES[1]}
SOURCE_VOCAB=${BPE_FILES[2]}
DEST_VOCAB=${BPE_FILES[3]}
echo "BPE Files:" $BPE_SOURCE $BPE_DEST $SOURCE_VOCAB $DEST_VOCAB
