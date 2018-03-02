#!/bin/bash

SRC_LANG=$1
DST_LANG=$2
SRC_MERGES=$3
DST_MERGES=$4
WIPE_MODELS=$5

module load anaconda/3-5.0.1 cuda/8.0.44 boost/1.58.0
source activate ethaconda

#echo $SRC_LANG $DST_LANG $SRC_MERGES $DST_MERGES

MOSES=/home/ualelm/mosesdecoder/scripts
MARIAN=/home/ualelm/marian/build
SCRIPTS=/home/ualelm/scripts
PATH=$SCRIPTS:$MOSES/recaser:$MOSES/tokenizer:$MARIAN:$PATH
echo "PATH =" $PATH

#python simple.py
#exit 0

EXPERIMENT=experiments/$SRC_LANG-$DST_LANG
DATA=$EXPERIMENT/data
VOCAB=$EXPERIMENT/vocab
TRANS=$EXPERIMENT/trans
MODEL=$EXPERIMENT/models/${SRC_LANG}-${DST_LANG}.npz
LOG=$EXPERIMENT/logs/${SRC_LANG}-${DST_LANG}.log
#echo $EXPERIMENT $DATA $VOCAB $TRANS $MODEL $LOG

TRAIN_SRC=$DATA/${SRC_LANG}.train
TRAIN_DST=$DATA/${DST_LANG}.train
#echo $TRAIN_SRC $TRAIN_DST

DEV_SRC=$DATA/${SRC_LANG}-src.dev.sgml
DEV_DST=$DATA/${DST_LANG}-ref.dev.sgml
DEV_SRC_PLAIN=$DATA/${SRC_LANG}.dev
#echo $DEV_SRC $DEV_DST $DEV_SRC_PLAIN

TST_SRC=$DATA/${SRC_LANG}-src.test.sgml
TST_DST=$DATA/${DST_LANG}-ref.test.sgml
TST_SRC_PLAIN=$DATA/${SRC_LANG}.test
#echo $TST_SRC $TST_DST $TST_SRC_PLAIN

SRC_CODES=$DATA/${SRC_LANG}.codes
DST_CODES=$DATA/${DST_LANG}.codes
#echo $SRC_CODES $DST_CODES

#exit 0


if [ $WIPE_MODELS ]
then
    rm $EXPERIMENT/models/*
fi

python experiment.py \
                        --train            $TRAIN_SRC $TRAIN_DST                \
                        --dev              $DEV_SRC   $DEV_DST   $DEV_SRC_PLAIN \
                        --test             $TST_SRC   $TST_DST   $TST_SRC_PLAIN \
                        --dest-lang        $DST_LANG                            \
                        --codes            $SRC_CODES $DST_CODES                \
                        --max-sequences    $SRC_MERGES $DST_MERGES              \
                        --translation-dir  $TRANS                               \
                        --vocab-dir        $VOCAB                               \
                        --train-log-prefix $LOG                                 \
                        --model-prefix     $MODEL                               \
                        --verbose
                       
