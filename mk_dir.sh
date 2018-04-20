#!/bin/bash
#set -x

SRC=$1
DST=$2

OLD_DIR=`pwd`

EXP=experiments/$SRC-$DST
mkdir -p $EXP

for directory in data models trans vocab logs
do
    mkdir $EXP/$directory
done

