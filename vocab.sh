#!/bin/bash

module load boost/1.58.0 cuda/8.0.44

DATA_DIR=$1
SOURCE_CORP=$2
DEST_CORP=$3
SUFFIX=$4

SOURCE_LANG=`basename -s $SUFFIX $SOURCE_CORP`
DEST_LANG=`basename -s $SUFFIX $DEST_CORP`

VOCAB=$DATA_DIR/${SOURCE_LANG}-${DEST_LANG}.yml

cat $SOURCE_CORP $DEST_CORP | \
  ~/marian/build/marian-vocab \
  > $VOCAB

echo $VOCAB
