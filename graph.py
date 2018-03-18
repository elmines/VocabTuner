import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import json
import sys
import os
import argparse

import numpy as np

def COLOR_A():
    return "blue"

def COLOR_B():
    return "red"

def create_parser():
    parser = argparse.ArgumentParser(description="Plot results of BPE-optimization experiment")
    parser.add_argument("--input", "-i", nargs="+", metavar="<path>", type=argparse.FileType("r"), help="1 to 2 JSON files containing an OptimizeResult object written as a dict") 
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


def __scatter(json_file, axes, color, source_lang, dest_lang):
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


    axes.scatter( source_merges[:-1], dest_merges[:-1],
                  c = color,
                  edgecolors = "black",
                  s = marker_sizes[:-1],
                  label = source_lang + " to " + dest_lang
    )


    #print(matplotlib.rcParams)

    
    axes.scatter( source_merges[-1], dest_merges[-1],
                  c = color,
                  edgecolors = "black",
                  s = marker_sizes[-1],
                  marker = "*",
                  linewidths = matplotlib.rcParams["lines.linewidth"] * 2,
                  label = "Best " + source_lang + " to " + dest_lang
    )

def gen_legend_handles(source_lang, dest_lang, bidir=False):
    
    handles = []
    handles.append( matplotlib.patches.Patch(color=COLOR_A(), label = source_lang + " to " + dest_lang) )
    if bidir:
        handles.append( matplotlib.patches.Patch(color=COLOR_B(), label = dest_lang + " to " + source_lang) )

    handles.append(plt.scatter([], [], label="Best pair",
                  c = "white",
                  edgecolors = "black",
                  s = 500.0,  
                  marker="*"
                  #linewidths = matplotlib.rcParams["lines.linewidth"] * 2
                  )
    )

    return handles
    
def graph_results(json_files, source_lang="xx", dest_lang="xx"):


    fig = plt.figure()
    axes = plt.subplot(1, 1, 1)

    #title = source_lang + "-->" + dest_lang if len(json_files) == 1 else source_lang + "<-->" + dest_lang
    title = source_lang + " and " + dest_lang
    axes.set_title(title)
    axes.set_xlabel(source_lang + " BPE merges")
    axes.set_ylabel(dest_lang + " BPE merges")

    __scatter(json_files[0], axes, COLOR_A(), source_lang, dest_lang)
    if len(json_files) > 1:
        __scatter(json_files[1], axes, COLOR_B(), dest_lang, source_lang)

    handles = gen_legend_handles(source_lang, dest_lang, len(json_files) > 1)

    #plt.legend(source_lang + " merges", dest_lang + " merges")

    axes.legend(handles = handles, loc = 3,
               bbox_to_anchor=(0.0, -0.15, 1.0, .102),
               ncol = len(handles),
               mode = "expand"
    )
    plt.show()
   

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    files = (args.input[0].name,) if len(args.input) == 1 else (args.input[0].name, args.input[1].name) 
    graph_results( files, args.langs[0], args.langs[1] )
