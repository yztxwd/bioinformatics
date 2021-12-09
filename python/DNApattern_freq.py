#!python3

description = """

    Compute the frequency of the given pattern on the input fasta file

    Example Usage:
        $ python DNApattern_freq.py pattern fasta

    Rquirements:
        pattern should be a regular expression can be recogonized by re module

    @Jianyu Yang, 2020

"""

import re, sys
import numpy as np

def main():
    # load the arguments
    if len(sys.argv != 3):
        print(description)
        raise Exception(f"Expect two arguments! Get {len(sys.argv)-1}")
    else:
        pattern, fasta = sys.argv[1:]

    # for each fasta record, convert it to the occurence of pattern array
    rp = re.compile(pattern)
    arrays = []
    with open(fasta, 'r') as f, open(fasta[:-len(".fa")], 'w') as o:
        for line in f:
            array = np.zeros(len(line.strip()))
            if line.startswith(">"):
                continue
            else:
                riter = rp.finditer(line.strip())
                for m in riter:
                    array[m.start(): m.end()] = 1
                o.write(",".join([str(i) for i in array]) + "\n")

if __name__ == "__main__":
    main()
