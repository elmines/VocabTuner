#!/bin/bash

SRC_LANG=$1
DST_LANG=$2
let OUT_DOMAIN=$3

DATA=$SRC_LANG-$DST_LANG/data

merge() {
   LANG_ABBREV=$1
   let OUT_SEQ=$2

   BIO=$DATA/${LANG_ABBREV}.bio.train
   OUT=$DATA/full-europarl.${LANG_ABBREV}
   MERGED=$DATA/${LANG_ABBREV}.train

   cat $BIO > $MERGED
   echo "" >> $MERGED
   head $OUT -n $OUT_SEQ >> $MERGED
   echo "Wrote $MERGED from $BIO and $OUT_SEQ sequences from $OUT"
}

merge $SRC_LANG $OUT_DOMAIN
merge $DST_LANG $OUT_DOMAIN
