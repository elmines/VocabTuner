#!/bin/bash

SRC_LANG=$1
DST_LANG=$2
SRC_MERGES=$3
DST_MERGES=$4
EXPERIMENTS_ROOT=$5
WIPE_MODELS=$6

module load cuda/8.0.44
module load anaconda/3-5.0.1 boost/1.58.0
source activate ethaconda

MOSES=/home/ualelm/mosesdecoder/scripts
MARIAN=/home/ualelm/marian/build
SCRIPTS=/home/ualelm/scripts
PATH=$SCRIPTS:$MOSES/recaser:$MOSES/tokenizer:$MARIAN:$PATH

EXPERIMENT=$EXPERIMENTS_ROOT/$SRC_LANG-$DST_LANG
DATA=$EXPERIMENT/data
VOCAB=$EXPERIMENT/vocab
TRANS=$EXPERIMENT/trans
MODEL=$EXPERIMENT/models/${SRC_LANG}-${DST_LANG}.npz
LOG=$EXPERIMENT/logs/${SRC_LANG}-${DST_LANG}.log

SRC_CODES=$DATA/${SRC_LANG}.codes
DST_CODES=$DATA/${DST_LANG}.codes
#Pass in the two above if using separate codes, the two below otherwise
JOINT_CODES=$DATA/$SRC_LANG-${DST_LANG}.codes 

TRAIN_SRC=$DATA/${SRC_LANG}.train.tok.tc
TRAIN_DST=$DATA/${DST_LANG}.train.tok.tc

DEV_SRC_SGML=$DATA/${SRC_LANG}-src.dev.sgml
DEV_DST_SGML=$DATA/${DST_LANG}-ref.dev.sgml
DEV_SRC=$DATA/${SRC_LANG}.dev.tok.tc
DEV_DST=$DATA/${DST_LANG}.dev.tok.tc

RESULTS=${SRC_LANG}-${DST_LANG}.json


if [ $WIPE_MODELS ]
then
    echo "Wiping models from" "${EXPERIMENT}/models"
    rm $EXPERIMENT/models/*
fi

python -u experiment.py \
                        --codes                $JOINT_CODES                \
                        --max-sequences        $SRC_MERGES   $DST_MERGES   \
                        --vocabulary-threshold 50                          \
                        --train                $TRAIN_SRC    $TRAIN_DST    \
                        --dev                  $DEV_SRC      $DEV_DST      \
                        --results              $RESULTS                    \
                        --dest-lang            $DST_LANG                   \
                        --translation-dir      $TRANS   --vocab-dir $VOCAB   --train-log-prefix $LOG   --model-prefix $MODEL \
                        --metric               "ce-mean-words"             \
                        --verbose

                        #--dev-sgml         $DEV_SRC_SGML $DEV_DST_SGML \
