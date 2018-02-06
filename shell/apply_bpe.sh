#!/bin/bash


RAW_CORP=$1
CODES=$2
CORP_VOCAB=$3

#Outputs
BPE_CORP=$4

#Constants (for now)
THRESH=50

if [ ! -e $BPE_CORP ]
then
    python3 ~/subword-nmt/apply_bpe.py \
    --codes $CODES \
    --vocabulary $CORP_VOCAB \
    --vocabulary-threshold $THRESH \
    < $RAW_CORP \
    > $BPE_CORP

fi

