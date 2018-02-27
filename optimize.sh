#!/bin/bash

SRC_LANG=$1
DST_LANG=$2
SRC_MERGES=$3
DST_MERGES=$4

MARIAN=/home/ualelm/marian/build
PATH=$MARIAN:$PATH:.

EXPERIMENT=experiments/$SRC_LANG-$DST_LANG
DATA=$EXPERIMENT/data
VOCAB=$EXPERIMENT/vocab
TRANS=$EXPERIMENT/trans
MODEL=$EXPERIMENT/models/${SRC_LANG}-${DST_LANG}.npz
LOG=$EXPERIMENT/logs/${SRC_LANG}-${DST_LANG}.log

TRAIN_SRC=$DATA/${SRC_LANG}.train
TRAIN_DST=$DATA/${DST_LANG}.train

DEV_SRC=$DATA/${SRC_LANG}-src.dev.sgml
DEV_DST=$DATA/${DST_LANG}-ref.dev.sgml
DEV_SRC_PLAIN=$DATA/${SRC_LANG}.dev

TST_SRC=$DATA/${SRC_LANG}-src.test.sgml
TST_DST=$DATA/${DST_LANG}-ref.test.sgml
TST_SRC_PLAIN=$DATA/${SRC_LANG}.test

SRC_CODES=$DATA/${SRC_LANG}.codes
DST_CODES=$DATA/${DST_LANG}.codes

module load anaconda/3-5.0.1 cuda/8.0.44 boost/1.58.0
source activate ethaconda

python experiment.py \
                        --train            $TRAIN_SRC $TRAIN_DST                \
                        --dev              $DEV_SRC   $DEV_DST   $DEV_SRC_PLAIN \
                        --test             $TST_SRC   $TST_DST   $TST_SRC_PLAIN \
                        --dst              $DST_LANG                            \
                        --codes            $SRC_CODES $DST_CODES                \
                        --max-sequences    $SRC_MERGES $DST_MERGES              \
                        --translation-dir  $TRANS                               \
                        --vocab-dir        $VOCAB                               \
                        --train-log-prefix $LOG                                 \
                        --model-prefix     $MODEL                               \
                        --verbose
                       
