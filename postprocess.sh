#!/bin/bash



DEST_LANG=$1
RAW_TRANSLATION=$2
SOURCE=$3
REFERENCE=$4

SYSTEM="TheUniversityOfAlabama"

SGML_TRANSLATION=${RAW_TRANSLATION}.sgm
#SGML_TRANSLATION=output.txt

cat $RAW_TRANSLATION | \
    sed -r 's/(@@ )|(@@ ?$)//g' | \
    ~/mosesdecoder/scripts/recaser/detruecase.perl | \
    ~/mosesdecoder/scripts/tokenizer/detokenizer.perl | \
    ~/scripts/wrap-xml.perl $DEST_LANG $SOURCE $SYSTEM \
    > $SGML_TRANSLATION
    
perl ~/scripts/mteval-v14.pl  -s $SOURCE -r $REFERENCE -t $SGML_TRANSLATION
