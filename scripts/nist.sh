#!/bin/bash

RAW_TRANSLATION=$1

SGML_TRANSLATION=${RAW_TRANSLATION}.sgm

SOURCE_SGM
REF_SGM

#Obtain file paths form SOURCE_SGM, REF_SGM, and TRANS_RAW



./postprocess.sh $RAW_TRANSLATION
python nist_score.py
