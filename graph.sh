#!/bin/bash

python3 graph.py --input results/ted/es-en-Mar6.json results/ted/en-es-Mar13.json --langs es en

python3 graph.py --input results/ted/es-fr-Mar10.json --langs es fr
python3 graph.py --input results/ted/de-en-Mar10.json --langs de en
python3 graph.py --input results/ted/de-ru-Mar13.json --langs de ru
