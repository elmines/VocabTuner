import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import json
import sys
import os
import argparse

import numpy as np

def graph_points(json_file):
    print("json_file: " + json_file)
    with open(json_file, "r") as f:
        res = json.load(f)

    domain = res["x_iters"]
    scores = res["func_vals"]
    print(res)
    #print(domain)
    #print(scores)

    source_merges = [ merge_pair[0] for merge_pair in domain ]
    dest_merges = [ merge_pair[1] for merge_pair in domain ]

    print(source_merges)
    print(dest_merges)
    print(scores)

    fig = plt.figure()
    axes = Axes3D(fig)
    axes.scatter( xs = np.asarray(source_merges), ys = np.asarray(dest_merges), zs = np.asarray(scores) )
    #Axes3D.scatter( xs = np.asarray(source_merges), ys = np.asarray(dest_merges), zs = np.asarray(scores) )

    

if __name__ == "__main__":
    json_file = os.path.abspath(sys.argv[1])
    graph_points(json_file)
