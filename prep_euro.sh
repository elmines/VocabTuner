#!/bin/bash

SRC=$1
DST=$2

EXP_ROOT=experiments

DATA_DIR=`pwd`/$EXP_ROOT/$SRC-$DST/data

ln -s $DATA_DIR/train.$SRC $DATA_DIR/${SRC}.orig
ln -s $DATA_DIR/train.$DST $DATA_DIR/${DST}.orig

ln -s $DATA_DIR/test.$SRC $DATA_DIR/${SRC}.test
ln -s $DATA_DIR/test.$DST $DATA_DIR/${DST}.test

./part.sh $SRC $DST $EXP_ROOT
./preproc.sh $SRC $DST $EXP_ROOT
