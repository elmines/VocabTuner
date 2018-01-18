#!/bin/bash

# $1 = Data directory
# $2 = Source Language
# $3 = Destination Language

DATA_DIR=$1
SOURCE_LANG=$2
DEST_LANG=$3

MOSES_SCRIPTS=/home/ualelm/mosesdecoder/scripts
BPE_SCRIPTS=/home/ualelm/subword-nmt

RAW_SOURCE=$DATA_DIR/${SOURCE_LANG}.train
RAW_DEST=$DATA_DIR/${DEST_LANG}.train

echo $RAW_SOURCE
echo $RAW_DEST

#TOK_SOURCE=`./tokenize $RAW_SOURCE`
#TOK_DEST=`./tokenize $RAW_DEST`

#echo $TOK_SOURCE
#echo $TOK_DEST


#$MOSES_SCRIPTS/tokenizer/tokenizer.perl 

#$MOSES_SCRIPTS/recaser/learn-truecase.perl

#$MOSES_SCRIPTS/recaser/truecase.perl




