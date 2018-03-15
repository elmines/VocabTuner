import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import json
import sys
import os
import argparse

import numpy as np


def create_parser():
    parser = argparse.ArgumentParser(description="Plot results of BPE-optimization experiment")
    parser.add_argument("--input", "-i", required=True, metavar="<path>", type=argparse.FileType("r"), help="JSON file containing OptimizeResult object written as a dict") 
    parser.add_argument("--langs", "-l", nargs=2, default=("xx", "xx"), help="Source and Destination languages")

    return parser

def linear_marker_sizes(scores, min_marker_size, max_marker_size):
    return np.asarray([min_marker_size + ( (max_marker_size - min_marker_size) * (score - min(scores)) / (max(scores) - min(scores)) ) for score in scores])

def log_marker_sizes(scores, min_marker_size, max_marker_size):
    min_score = min(scores)
    max_score = max(scores)
    marker_sizes = np.ndarray(shape=(len(scores),), dtype=scores.dtype)
    for i in range(len(scores)):
        marker_sizes[i] = min_marker_size - 1 + pow(max_marker_size - min_marker_size + 1, (scores[i] - min_score)/(max_score - min_score))
                          
    return marker_sizes


def graph_points(json_file, source_lang="xx", dest_lang="xx"):
    with open(json_file, "r") as f:
        res = json.load(f)

    domain = res["x_iters"]
    scores = np.asarray( res["func_vals"] )

    source_merges = np.asarray( [ merge_pair[0] for merge_pair in domain ] )
    dest_merges = np.asarray( [ merge_pair[1] for merge_pair in domain ] )

    #Actual BLEU scores
    scores = 100 - scores

    #Sort scores in ascending order
    ind = np.unravel_index(np.argsort(scores, axis=None), scores.shape)
    source_merges = source_merges[ind]
    dest_merges = dest_merges[ind]
    scores = scores[ind]

    #Let marker sizes be a function of the score
    min_marker_size = 6.0
    max_marker_size = 1000.0
    marker_sizes = log_marker_sizes(scores, min_marker_size, max_marker_size)

    fig = plt.figure()
    axes = plt.axes()
    axes.set_title("%s-->%s" % (source_lang, dest_lang))
    axes.set_xlabel(source_lang)
    axes.set_ylabel(dest_lang)

    axes.scatter( source_merges, dest_merges,
                  edgecolors = "black",
                  s = marker_sizes
    )

    plt.show()

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    graph_points(  args.input.name, args.langs[0], args.langs[1] )
