#!/bin/bash



DEST_LANG=$1
RAW_TRANSLATION=$2
SOURCE=$3
REFERENCE=$4


cat $RAW_TRANSLATION | \
    sed -r 's/(@@ )|(@@ ?$)//g' | \
    ~mosesdecoder/scripts/recaser/detruecase.perl | \
    ~mosesdecoder/scripts/tokenizer/detokenizer.perl | \
    ~/scripts/wrap-xml.perl $DEST_LANG $REFERENCE | \
    > $SGML_TRANSLATION
    
perl ~/scripts/mteval-v14.perl -r $REFERENCE -s $SOURCE -t $SGML_TRANSLATION
