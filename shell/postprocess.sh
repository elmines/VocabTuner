#!/bin/bash

#Outputs text wrapped in SGML to stdout




DEST_LANG=$1
SOURCE=$2


#Constants
SYSTEM="TheUniversityOfAlabama"


sed -r 's/(@@ )|(@@ ?$)//g' | \
    ~/mosesdecoder/scripts/recaser/detruecase.perl | \
    ~/mosesdecoder/scripts/tokenizer/detokenizer.perl | \
    ~/scripts/wrap-xml.perl $DEST_LANG $SOURCE $SYSTEM
