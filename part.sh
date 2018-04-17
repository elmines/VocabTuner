#!/bin/bash
SRC_LANG=$1
DST_LANG=$2
EXPERIMENTS_ROOT=$3

DATA=$EXPERIMENTS_ROOT/$SRC_LANG-$DST_LANG/data
ORIG_SRC=$DATA/${SRC_LANG}.orig
ORIG_DST=$DATA/${DST_LANG}.orig

CORP_NAME=europarl-v7

python3 part.py \
                    --ratio            299:1:0                                                 \
                    --max-sequences    1500000                                                 \
                    --input            $ORIG_SRC                       $ORIG_DST               \
                    --train            $DATA/${SRC_LANG}.train         $DATA/${DST_LANG}.train \
                    --dev_id           ${CORP_NAME}.custom-dev                                 \
                    --test_id          ${CORP_NAME}.custom-test                                \
                    --dev_plain        $DATA/${SRC_LANG}.dev           $DATA/${DST_LANG}.dev   \
                    --trglang          ${DST_LANG}                                             \
                    --verbose

#--test_plain       $DATA/${SRC_LANG}.test $DATA/${DST_LANG}.test \
#--dev              $DATA/${SRC_LANG}-src.dev.sgml  $DATA/${DST_LANG}-ref.dev.sgml \
#--test             $DATA/${SRC_LANG}-src.test.sgml $DATA/${DST_LANG}-ref.test.sgml \
