#!/bin/bash

SRC_LANG=$1
DST_LANG=$2
EXPERIMENTS_ROOT=$3

#Add Moses preprocessing scripts to PATH
MOSES=/home/ualelm/mosesdecoder
TOKENIZATION=$MOSES/scripts/tokenizer
CASING=$MOSES/scripts/recaser
PATH=$TOKENIZATION:$CASING:$PATH

DATA=$EXPERIMENTS_ROOT/$SRC_LANG-$DST_LANG/data
TRAIN_SRC=$DATA/${SRC_LANG}.train
TRAIN_DST=$DATA/${DST_LANG}.train

DEV_SRC=$DATA/${SRC_LANG}.dev
DEV_DST=$DATA/${DST_LANG}.dev

TST_SRC=$DATA/${SRC_LANG}.test
TST_DST=$DATA/${DST_LANG}.test

let MAX_CODES=200000

python3 preprocess.py  --train $TRAIN_SRC $TRAIN_DST \
                      --langs $SRC_LANG $DST_LANG \
                      --num-sequences $MAX_CODES \
                      --joint \
                      --write-dir $DATA \
                      --extra-source $DEV_SRC $TST_SRC \
                      --extra-dest   $DEV_DST $TST_DST \
                      --verbose
