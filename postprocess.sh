#!/bin/bash

#Outputs text wrapped in SGML to stdout




RAW_TRANSLATION=$1
DEST_LANG=$2
SOURCE=$3


#Constants
SYSTEM="TheUniversityOfAlabama"


cat $RAW_TRANSLATION | \
    sed -r 's/(@@ )|(@@ ?$)//g' | \
    ~/mosesdecoder/scripts/recaser/detruecase.perl | \
    ~/mosesdecoder/scripts/tokenizer/detokenizer.perl | \
    ~/scripts/wrap-xml.perl $DEST_LANG $SOURCE $SYSTEM
