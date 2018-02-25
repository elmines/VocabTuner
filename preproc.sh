#!/bin/bash

SRC_LANG=es
DST_LANG=en

MOSES=/home/ualelm/mosesdecoder
TOKENIZATION=$MOSES/scripts/tokenizer
CASING=$MOSES/scripts/recaser

PATH=$TOKENIZATION:$CASING:$PATH
#echo "New path =" $PATH


DATA=experiments/$SRC_LANG-$DST_LANG/data
TRAIN_SRC=$DATA/${SRC_LANG}.train
TRAIN_DST=$DATA/${DST_LANG}.train

JOINT_CODES=$DATA/${SRC_LANG}-${DST_LANG}.codes
SRC_CODES=$DATA/${SRC_LANG}.codes
DST_CODES=$DATA/${DST_LANG}.codes

python3 preprocess.py --train $TRAIN_SRC $TRAIN_DST \
                      --codes $SRC_CODES $DST_CODES \
                      --langs $SRC_LANG $DST_LANG \
                      --write-dir $DATA 
                      
                      #--joint 

