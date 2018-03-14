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

def graph_points(json_file, source_lang="xx", dest_lang="xx"):
    with open(json_file, "r") as f:
        res = json.load(f)

    domain = res["x_iters"]
    scores = np.asarray( res["func_vals"] )

    source_merges = np.asarray( [ merge_pair[0] for merge_pair in domain ] )
    dest_merges = np.asarray( [ merge_pair[1] for merge_pair in domain ] )

    #Actual BLEU scores
    scores = 100 - scores

    ind = np.unravel_index(np.argsort(scores, axis=None), scores.shape)
    source_merges = source_merges[ind]
    dest_merges = dest_merges[ind]
    scores = scores[ind]

    #matplotlib.rcParams["lines.markersize"] = 30.0
    min_marker_size = 6.0
    max_marker_size = 1000.0

    #marker_sizes = np.ndarray(shape=(len(source_merges),), dtype=source_merges.dtype)

   

    marker_sizes = np.asarray([min_marker_size + ( (max_marker_size - min_marker_size) * (score - min(scores)) / (max(scores) - min(scores)) ) for score in scores])
    #print(marker_sizes)
    fig = plt.figure()

    axes = plt.axes()
    #axes = plt.axes(projection="3d")
    axes.set_title("%s-->%s" % (source_lang, dest_lang))
    axes.set_xlabel(source_lang)
    axes.set_ylabel(dest_lang)
    #axes.set_zlabel("BLEU")

    axes.scatter( source_merges, dest_merges,
                  edgecolors = "black",
                  s = marker_sizes
                  #, s = scores)
    )

    plt.show()

def practicing():
    def f(x, y):
        return np.sin(np.sqrt(x ** 2 + y ** 2))

    x = np.linspace(-6, 6, 6)
    y = np.linspace(-4, 4, 4)
    print(x)
    print(y)

    X, Y = np.meshgrid(x, y)
    Z = f(X, Y) 
    print(X)
    print(Y)
    print(Z)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z');

    plt.show()

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    graph_points(  args.input.name, args.langs[0], args.langs[1] )
