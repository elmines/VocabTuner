#!/bin/bash

CORPUS=$1

TOK_CORPUS=${CORPUS}.tok

/home/ualelm/mosesdecoder/scripts/tokenizer/tokenizer.perl \
  < $CORPUS \
  > $TOK_CORPUS

echo $TOK_CORPUS
