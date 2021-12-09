#!/usr/bin/env python
import sys
import numpy as np
import pyBigWig

description = """

    Given chromosome sizes, create a new bigwig file with all entries in the original bigwig file.
    This is to ensure deeptools computeMatrix won't skip chromosome when the queried chromosome is not included in bigwig file

    Warning: This script is not appropriate for tense signal track, it would take several hours to write each base pair into output

    Usage:
        
        $ python fixBigWig.py mm10.chrom.sizes input.bw output.bw

    Reference:
        https://bioinformatics.stackexchange.com/questions/2729/adding-entries-to-bigwig-file

"""


chromsizes, ibw, obw = sys.argv[1:]

ifile = pyBigWig.open(ibw)
ofile = pyBigWig.open(obw, "w")

# Create/Add the header
hdr = []
with open(chromsizes, "r") as f:
    for line in f:
        chrom, size = line.strip().split("\t")[:2]
        size = int(size)
        hdr.append((chrom, size))
ofile.addHeader(hdr)

# Write the bigWig
step = 10000000
for _ in hdr:
    start = 0
    while start < _[1]:
        end = (start + step) if (start + step) <= (_[1]) else _[1]
        try:
            ints = ifile.intervals(_[0], start, end)
            starts = [x[0] for x in ints]
            ends = [x[1] for x in ints]
            vals = [x[2] for x in ints]
            ofile.addEntries([_[0]] * len(starts), starts, ends=ends, values=vals)
        except:
            # intervals() will throw an error if the chromosome isn't present
            pass

        start = end
ofile.close()
ifile.close()
