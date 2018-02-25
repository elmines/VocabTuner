#!/bin/bash
#set -x

SRC=$1
DST=$2

SOURCE_DIR=~/data/ted/2012-release/
OLD_DIR=`pwd`

cd $SOURCE_DIR
tar -xzvf $SOURCE_DIR/${SRC}-${DST}.tgz
cd $OLD_DIR

EXP=experiments/$SRC-$DST
mkdir $EXP

for directory in data models trans vocab logs
do
    mkdir $EXP/$directory
done

ln -s $SOURCE_DIR/${SRC}-${DST}/train.$SRC-${DST}.$SRC $EXP/data/${SRC}.orig
ln -s $SOURCE_DIR/${SRC}-${DST}/train.$SRC-${DST}.$DST $EXP/data/${DST}.orig
