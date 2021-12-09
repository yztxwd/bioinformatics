#!/usr/bin/env python
import sys
import numpy as np
import pyBigWig

description = """

    Given chromosome sizes, create a new bigwig file with all entries in the original bedGraph file
    This is to ensure deeptools computeMatrix won't skip chromosome when the queried chromosome is not included in bigwig file

    Warning: This script is not appropriate for tense signal track, it would take several hours to write each base pair into output

    Usage:
        
        $ python bedGraphToBigWig.py mm10.chrom.sizes input.bedGraph output.bw

    Reference:
        https://bioinformatics.stackexchange.com/questions/2729/adding-entries-to-bigwig-file

"""


chromsizes, ibg, obw = sys.argv[1:]

ifile = open(ibg, "r")
ofile = pyBigWig.open(obw, "w")

# Create/Add the header
hdr = []
with open(chromsizes, "r") as f:
    for line in f:
        chrom, size = line.strip().split("\t")[:2]
        size = int(size)
        hdr.append((chrom, size))
ofile.addHeader(hdr, maxZooms=10)

# Write the bigWig
lastChrom = None
starts = []
ends = []
vals = []
for line in ifile:
    interval = line.split()
    # Buffer up to a million entries
    if interval[0] != lastChrom or len(starts) == 1000000:
        if lastChrom is not None:
            ofile.addEntries([lastChrom] * len(starts), starts, ends=ends, values=vals)
        lastChrom = interval[0]
        starts = [int(interval[1])]
        ends = [int(interval[2])]
        vals = [float(interval[3])]
    else:
        starts.append(int(interval[1]))
        ends.append(int(interval[2]))
        vals.append(float(interval[3]))
if len(starts) > 0:
    ofile.addEntries([lastChrom] * len(starts), starts, ends=ends, values=vals)
ifile.close()
ofile.close()
