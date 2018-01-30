#!/bin/bash

#Simple wrapper script to use mteval-v14.pl with my command line options
#Outputs results to stdout

SOURCE=$1
REFERENCE=$2
TRANSLATON=$3

perl ~/scripts/mteval-v14.pl  -s $SOURCE -r $REFERENCE -t $TRANSLATION \
    -n \
    --metricsMATR \
    -c
