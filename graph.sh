#!/bin/bash

python3 graph.py --input         results/ted/es-en-Mar6.json \
                                     results/ted/en-es-Mar13.json \
                                     results/ted/es-fr-Mar10.json \
                                     results/ted/de-en-Mar10.json \
                                     results/ted/de-ru-Mar13.json \
                 --indices       1 \
                                     1 \
                                     2 \
                                     3 \
                                     4 \
                 --langs         Spanish-English Spanish-French German-English German-Russian

#python3 graph.py --input results/ted/es-fr-Mar10.json --langs es fr
#python3 graph.py --input results/ted/de-en-Mar10.json --langs de en
#python3 graph.py --input results/ted/de-ru-Mar13.json --langs de ru
