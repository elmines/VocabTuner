#!/bin/bash

SRC_LANG=$1
DST_LANG=$2
SRC_MERGES=$3
DST_MERGES=$4
WIPE_MODELS=$5

module load anaconda/3-5.0.1 cuda/8.0.44 boost/1.58.0
source activate ethaconda

MOSES=/home/ualelm/mosesdecoder/scripts
MARIAN=/home/ualelm/marian/build
SCRIPTS=/home/ualelm/scripts
PATH=$SCRIPTS:$MOSES/recaser:$MOSES/tokenizer:$MARIAN:$PATH

EXPERIMENT=experiments/$SRC_LANG-$DST_LANG
DATA=$EXPERIMENT/data
VOCAB=$EXPERIMENT/vocab
TRANS=$EXPERIMENT/trans
MODEL=$EXPERIMENT/models/${SRC_LANG}-${DST_LANG}.npz
LOG=$EXPERIMENT/logs/${SRC_LANG}-${DST_LANG}.log

SRC_CODES=$DATA/${SRC_LANG}.codes
DST_CODES=$DATA/${DST_LANG}.codes

TRAIN_SRC=$DATA/${SRC_LANG}.train
TRAIN_DST=$DATA/${DST_LANG}.train

DEV_SRC_SGML=$DATA/${SRC_LANG}-src.dev.sgml
DEV_DST_SGML=$DATA/${DST_LANG}-ref.dev.sgml
DEV_SRC=$DATA/${SRC_LANG}.dev
DEV_DST=$DATA/${DST_LANG}.dev

RESULTS=${SRC_LANG}-${DST_LANG}.json


if [ $WIPE_MODELS ]
then
    rm $EXPERIMENT/models/*
fi

python experiment.py \
                        --codes            $SRC_CODES    $DST_CODES    \
                        --max-sequences    $SRC_MERGES   $DST_MERGES   \
                        --train            $TRAIN_SRC    $TRAIN_DST    \
                        --dev              $DEV_SRC                    \
                        --dev-sgml         $DEV_SRC_SGML $DEV_DST_SGML \
                        --results          $RESULTS                    \
                        --dest-lang        $DST_LANG                   \
                        --translation-dir  $TRANS   --vocab-dir $VOCAB   --train-log-prefix $LOG   --model-prefix $MODEL \
                        --verbose
