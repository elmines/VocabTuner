import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import json
import sys
import os
import argparse

import numpy as np

def SUP_TITLE_FONT():
    return 32

def SUB_TITLE_FONT():
    return SUP_TITLE_FONT() * 3 / 4
    #return SUP_TITLE_FONT() * 3 / 4

def MINOR_FONT():
    return SUP_TITLE_FONT() * 5 / 8

def TICK_FONT():
    return SUP_TITLE_FONT() / 2

def LEGEND_FONT():
    return TICK_FONT()

def COLOR_A():
    return "blue"

def COLOR_B():
    return "red"

def create_parser():
    parser = argparse.ArgumentParser(description="Plot results of BPE-optimization experiment")
    parser.add_argument("--input", "-i", nargs="+", required=True, metavar="<path>", type=argparse.FileType("r"), help="1+ JSON files containing an OptimizeResult object written as a dict") 
    parser.add_argument("--langs", "-l", nargs="+", default=["xx"], help="Source and Destination language pairs (format: <source>-<dest>)")

    parser.add_argument("--indices", nargs="+", type=int, help="Pyplot subplot indices (number of unique indices must equal len(--langs))")

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
                  c = "white" if bidir else COLOR_A(),
                  edgecolors = "black",
                  s = 500.0,  
                  marker="*"
                  #linewidths = matplotlib.rcParams["lines.linewidth"] * 2
                  )
    )

    return handles
    
def graph_results(json_files, axes, source_lang="xx", dest_lang="xx"):

    if len(json_files) > 1:
        title = source_lang + " and " + dest_lang + ", Both Ways"
    else:
        title = source_lang + " to " + dest_lang
    axes.set_title(title, fontsize = SUB_TITLE_FONT())
    axes.set_xlabel(source_lang + " BPE merges", fontsize=MINOR_FONT())
    axes.set_ylabel(dest_lang + " BPE merges", fontsize=MINOR_FONT())
    axes.tick_params(labelsize=TICK_FONT())

    __scatter(json_files[0], axes, COLOR_A(), source_lang, dest_lang)
    if len(json_files) > 1:
        __scatter(json_files[1], axes, COLOR_B(), dest_lang, source_lang)

    handles = gen_legend_handles(source_lang, dest_lang, len(json_files) > 1)


    axes.legend(handles = handles, loc = 3,
               bbox_to_anchor=(0.0, -0.30, 1.0, .102),
               ncol = 2,
               mode = "expand"
               , fontsize=LEGEND_FONT()
    )
    #plt.show()


def ascending(items):
    return items == sorted(items)

def check_lengths(input, langs, indices):
    #lengths = [len(args.input), len(args.langs)]

    if len(input) != len(indices):
        raise ValueError("--input and --indices must have the same number of args") 
    elif len(args.langs) != len( set(indices) ):
        raise ValueError("--langs must have as many arguments as there are unique --indices")

    #first_len = lengths[0]
    #for length in lengths[1:]:
       #if first_len != length:
          #raise ValueError("--input, --langs, and --indices (if applicable) must have the same number of args")

def check_indices(indices):
    if not ascending(args.indices):
       raise ValueError("--indices must be in ascending order")
    if indices[0] != 1:
       raise ValueError("--indices must start at 1")
    for i in range( min(indices), max(indices) + 1 ):
        if i not in indices:
            raise ValueError("--indices must have all indices in the interval [%d, %d]" % (min(indices), max(indices)) )

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    indices = args.indices if args.indices else [ i + 1 for i in range(len(args.input)) ]
    check_lengths(args.input, args.langs, indices)

    fig = plt.figure( figsize=(21.5, 17),
                      #figsize = (12, 12),
                      dpi=128
    )
    fig.suptitle("BLEU Translation Scores by Size", fontsize = SUP_TITLE_FONT(), weight="bold")

    ncols = 2
    nrows = indices[-1] // ncols if (indices[-1] % ncols == 0) else (indices[-1] // ncols + indices[-1] % ncols)

    i = 0
    last_index = -1
    while i < len(args.input):
        file_set = [args.input[i].name]
        index = indices[i]
        lang_pair = args.langs[index - 1].split("-")

        while (i + 1 < len(args.input)) and (indices[i + 1] == index) :
            i += 1 
            file_set.append( args.input[i].name )

        #Generate new set of axes for new plot
        if index != last_index: axes = plt.subplot(nrows, ncols, index)

        graph_results(file_set, axes, lang_pair[0], lang_pair[1])
        i += 1

    #plt.tight_layout(pad = 1.08, h_pad = 1.08, w_pad = 1.08)
    plt.subplots_adjust(wspace = 0.3, hspace = 0.4)
    plt.savefig("/mnt/c/Users/elmin/Downloads/test.png")
    plt.savefig("test.png")
    #plt.show()
