#!python3

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, Value

description = """

    Plot v-plot which describes the fragment size && coverage information in distinct region

    Example uage:
        $ python vPlot.py intersect.bed threads output

    Requirements:
        input intersect.bed should be output of bedtools intersect with -wa -wb option, which will describe the input region info and overlapped fragment info

"""

class Counter(object):
    def __init__(self, init=0):
        self.val = Value("i", 0)
    
    def increment(self):
        with self.val.get_lock():
            self.val.value += 1
        print("%s0000 lines have been handled..." % self.val.value)

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

def dump_matrix(input):
    chunk, counter = input

    # initialize v-plot matrix
    rl = chunk.loc[1, "a_end"] - chunk.loc[1, "a_start"]
    vMatrix = np.zeros([1000, rl])

    # for each record, add the fragment size info into the v-plot matrix
    count = 0
    for index in chunk.index:
        a_start, a_end, b_start, b_end, length = chunk.loc[index, ["a_start", "a_end", "b_start", "b_end", "b_length"]]
        o_start, o_end = find_overlap(a_start, a_end, b_start, b_end)
        vMatrix[length-1, o_start: o_end] += 1
        count += 1
        if count % 10000 == 0:
            counter.increment()
    
    return vMatrix

def main():
    # load parameters
    if len(sys.argv) != 4:
        print(description)
        raise Exception("Expected 3 parameters!")
    else:
        bed, threads, output = sys.argv[1:]
        threads = int(threads)

    # load intersect bed file
    chunks = pd.read_table(bed, header=None, comment="#", names=["a_chr", "a_start", "a_end", "b_chr", "b_start", "b_end", "b_name", "b_length"],
                        chunksize = 10**6)

    # multiprocess
    ## start 
    ps = []
    p = Pool(threads)
    counter = Counter()
    for chunk in chunks:
        ps.append(p.apply_async(dump_matrix, args = ((chunk, counter), )))
    p.close()
    p.join()

    ## get results
    results = []
    for process in ps:
        results.append(process.get())

    # add up all results
    for index in range(len(results)):
        if index == 0:
            vMatrix = results[index]
        else:
            vMatrix += results[index]
    
    # save the v-plot matrix
    np.savetxt("%s.txt" % output, vMatrix)

    # get v-plot
#    plt.figure(figsize=(15, 15))
#    plt.

if __name__ == "__main__":
    main()
