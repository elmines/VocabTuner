#!/bin/bash

module load boost/1.58.0 cuda/8.0.44


#Inputs
MODEL=$1
SOURCE_VOCAB=$2
DEST_VOCAB=$3
SOURCE_TEXT=$4

~/marian/build/marian-decoder \
   --models $MODEL \
   --input $SOURCE_TEXT \
   --vocabs $SOURCE_VOCAB $DEST_VOCAB \
   --devices 0


