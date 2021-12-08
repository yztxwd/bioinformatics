#!/usr/bin/env python
import sys
import traceback
import pyBigWig

description = """

    Given chromosome sizes, create a new bigwig file with all entries in the original bigwig file.
    This is to ensure deeptools computeMatrix won't skip chromosome when the queried chromosome is not included in bigwig file

    Usage:
        
        $ python fixBigWig.py mm10.chrom.sizes input.bw output.bw

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
for _ in hdr:
    try:
        ints = ifile.intervals(_[0])
        starts = [x[0] for x in ints]
        ends = [x[1] for x in ints]
        vals = [x[2] for x in ints]
        ofile.addEntries([_[0]] * len(starts), starts, ends=ends, values=vals)
    except:
        print(traceback.format_exc())
        # intervals() will throw an error if the chromosome isn't present
        pass
ofile.close()
ifile.close()
