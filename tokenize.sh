#!/bin/bash

#set -x
CORPUS=$1
SUFFIX=$2
LANG_ABBR=$3

#Outputs file path of tokenized CORPUS
TOK_CORPUS=${CORPUS}.${SUFFIX}

if [ ! -e $TOK_CORPUS ]
then
  /home/ualelm/mosesdecoder/scripts/tokenizer/tokenizer.perl -l $LANG_ABBR -threads 4 < $CORPUS > $TOK_CORPUS
fi

#echo $TOK_CORPUS
