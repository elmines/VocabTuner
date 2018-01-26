#!/bin/bash

####PREPROCESS.SH############
#Preprocess the training, validation, and testing data for a given MT experiment
#Names of raw files must be in the following format:
#    Training: {LANGUAGE}.train
#    Development: {LANGUAGE}.dev
#    Testing: {LANGUAGE}.test
#
#Additional suffixes are appended to the generated, preprocessed text files.
#For example, after a Spanish training corpus has been tokenized, truecased,
#and split into BPE-subwords, the generated file would be:
#    es.train.tok.tc.bpe


# $1 = Data directory
# $2 = Source Language
# $3 = Destination Language

#Within the data directory, there should be a directory {Source Language}-{Destination Language}

DATA_ROOT=$1
SOURCE_LANG=$2
DEST_LANG=$3

DATA_DIR=$DATA_ROOT/${SOURCE_LANG}-${DEST_LANG}
if [ ! -e $DATA_DIR ]
then
  DATA_DIR=$DATA_ROOT/${DEST_LANG}-${SOURCE_LANG}
  if [ ! -e $DATA_DIR ]
  then
    echo "Error: no data for language pair ${SOURCE_LANG}-${DEST_LANG}"
    exit 1
  fi
fi

TRAIN_SUFFIX=.train
#ROLES=($TRAIN_SUFFIX .dev .test)
ROLES=($TRAIN_SUFFIX .dev)


RUNNING_SUFFIX=""
TOK_SUFFIX=.tok
for ROLE in ${ROLES[*]}
do
    ./tokenize.sh $DATA_DIR/${SOURCE_LANG}.$ROLE $TOK_SUFFIX $SOURCE_LANG
    ./tokenize.sh $DATA_DIR/${DEST_LANG}.$ROLE   $TOK_SUFFIX $DEST_LANG
done
RUNNING_SUFFIX=${RUNNING_SUFFIX}.$TOK_SUFFIX

SOURCE_TC_MODEL=${SOURCE_LANG}.model
DEST_TC_MODEL=${DEST_LANG}.model
./learn_truecase.sh \
     $DATA_DIR/${SOURCE_LANG}.$TRAIN_SUFFIX.$RUNNING_SUFFIX \
     $SOURCE_TC_MODEL
./learn_truecase.sh \
     $DATA_DIR/${DEST_LANG}.$TRAIN_SUFFIX.$RUNNING_SUFFIX \
     $DEST_TC_MODEL

TC_SUFFIX=.tc
for ROLE in ${ROLES[*]}
do
    RAW_SOURCE=$DATA_DIR/${SOURCE_LANG}.$ROLE.$RUNNING_SUFFIX
    TC_SOURCE=${RAW_SOURCE}.$TC_SUFFIX

    RAW_DEST=$DATA_DIR/${DEST_LANG}.$ROLE.$RUNNING_SUFFIX
    TC_DEST=${RAW_DEST}.$TC_SUFFIX

    ./truecase.sh $SOURCE_TC_MODEL $RAW_SOURCE $TC_SOURCE
    ./truecase.sh $DEST_TC_MODEL   $RAW_DEST   $TC_DEST
done

SOURCE_VOCAB=$DATA_DIR/${SOURCE_LANG}.vocab
DEST_VOCAB=$DATA_DIR/${DEST_LANG}.vocab
#./learn_bpe.sh

RUNNING_SUFFIX=${RUNNING_SUFFIX}.$TC_SUFFIX
BPE_SUFFIX=.bpe
for ROLE in ${ROLES[*]}
do
    ./apply_bpe.sh $DATA_DIR/${SOURCE_LANG}.$ROLE.$RUNNING_SUFFIX
    ./apply_bpe.sh $DATA_DIR/${DEST_LANG}.$ROLE.$RUNNING_SUFFIX
done

RUNNING_SUFFIX=${RUNNING_SUFFIX}.$BPE_SUFFIX

BPE_SOURCE=${BPE_FILES[0]}
BPE_DEST=${BPE_FILES[1]}

SUFFIX=.train.tok.tc.bpe
JOINT_VOCAB=`./vocab.sh $DATA_DIR $BPE_SOURCE $BPE_DEST $SUFFIX`
echo "Joint Vocab File:" $JOINT_VOCAB
