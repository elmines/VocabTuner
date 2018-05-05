#!/bin/bash
#set -x

ROOT=$1
SRC=$2
DST=$3

OLD_DIR=`pwd`

EXP=$ROOT/$SRC-$DST
mkdir -p $EXP

for directory in data models trans vocab logs
do
    mkdir $EXP/$directory
done

