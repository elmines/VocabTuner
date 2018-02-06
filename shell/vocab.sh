#!/bin/bash

module load boost/1.58.0 cuda/8.0.44

SOURCE_CORP=$1
DEST_CORP=$2

#Outputs
VOCAB=$3

if [ ! -e $VOCAB ]
then
   cat $SOURCE_CORP $DEST_CORP | \
     ~/marian/build/marian-vocab \
     > $VOCAB
fi
