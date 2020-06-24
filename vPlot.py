#!python3

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

description = """

    Plot v-plot which describes the fragment size && coverage information in distinct region

    Example uage:
        $ python vPlot.py intersect.bed output

    Requirements:
        input intersect.bed should be output of bedtools intersect with -wa -wb option, which will describe the input region info and overlapped fragment info

"""

def find_overlap(a_start, a_end, b_start, b_end):
    """
        find the relative coordinate of the overlapped region
    """

    if b_start > a_end or b_end < a_start:
        raise Exception("Non-overlapped record detected!")
    
    if b_start < a_start:
        o_start = 0
    else:
        o_start = b_start - a_start
    
    if b_end < a_end:
        o_end = b_end - a_start
    else:
        o_end = a_end - a_start
    
    return o_start, o_end

def main():
    # load parameters
    if len(sys.argv) != 3:
        print(description)
        raise Exception("Expected 2 parameters!")
    else:
        bed, output = sys.argv[1:]

    # load intersect bed file
    df = pd.read_table(bed, header=None, comment="#")
    df.columns = ["a_chr", "a_start", "a_end", "b_chr", "b_start", "b_end", "b_name", "b_length"]

    # initialize v-plot matrix
    rl = df.loc[1, "r_end"] - df.loc[1, "r_start"]
    vMatrix = np.array(1000, rl)

    # for each record, add the fragment size info into the v-plot matrix
    for index in df.index:
        a_start, a_end, b_start, b_end, length = df.loc[index, ["a_start", "a_end", "b_start", "b_end", "b_length"]]
        o_start, o_end = find_overlap(a_start, a_end, b_start, b_end)
        vMatrix[length-1, o_start: o_end] += 1
    
    # save the v-plot matrix
    np.savetxt("%s.txt" % output, vMatrix)

    # get v-plot
#    plt.figure(figsize=(15, 15))
#    plt.

if __name__ == "__main__":
    main()
