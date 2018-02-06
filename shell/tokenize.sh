#!/bin/bash

LANG_ABBR=$1
CORPUS=$2
TOK_CORPUS=$3

#FIXME: Abstract threads to a parameter or something
NUM_THREADS=4

if [ ! -e $TOK_CORPUS ]
then
  /home/ualelm/mosesdecoder/scripts/tokenizer/tokenizer.perl -l $LANG_ABBR -threads $NUM_THREADS < $CORPUS > $TOK_CORPUS
fi

