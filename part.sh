#!/bin/bash
SRC_LANG=$1
DST_LANG=$2

DATA=experiments/$SRC_LANG-$DST_LANG/data
TRAIN_SRC=$DATA/${SRC_LANG}.orig
TRAIN_DST=$DATA/${DST_LANG}.orig

CORP_NAME=ted2012

python3 part.py \
                    --ratio            20:1:1 \
                    --max-sequences    110000 \
                    --input            $DATA/${SRC_LANG}.orig          $DATA/${DST_LANG}.orig \
                    --train            $DATA/${SRC_LANG}.train         $DATA/${DST_LANG}.train \
                    --dev              $DATA/${SRC_LANG}-src.dev.sgml  $DATA/${DST_LANG}-ref.dev.sgml \
                    --test             $DATA/${SRC_LANG}-src.test.sgml $DATA/${DST_LANG}-ref.test.sgml \
                    --dev_id           ${CORP_NAME}.custom-dev \
                    --test_id          ${CORP_NAME}.custom-test \
                    --dev_plain        $DATA/${SRC_LANG}.dev \
                    --test_plain       $DATA/${SRC_LANG}.test \
                    --trglang          ${DST_LANG}      \
                    --verbose

                    
