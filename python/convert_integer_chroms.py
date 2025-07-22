#!/usr/bin/env python3

DESCRIPTION = """
Convert Integer chromosome names to roman numerals, assume the first column is the chromosome name in format "chr1", "chr2", ..., "chr10", etc. Note chrX is treated as chr10, there is no X, Y, M chromosome assumed
"""

from argparse import ArgumentParser

# Python3 program to convert
# integer value to roman values


# Function to convert integer to Roman values
def returnRoman(number):
    num = [1, 4, 5, 9, 10, 40, 50, 90, 100, 400, 500, 900, 1000]
    sym = ["I", "IV", "V", "IX", "X", "XL", "L", "XC", "C", "CD", "D", "CM", "M"]
    i = 12

    numeral = ""
    while number:
        div = number // num[i]
        number %= num[i]

        while div:
            numeral += sym[i]
            div -= 1
        i -= 1

    return numeral


if __name__ == "__main__":
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "inpt_bed",
        type=str,
        help="Input BED file with chromosome names in Integer format",
    )
    parser.add_argument(
        "output_bed",
        type=str,
        help="Output BED file with chromosome names in Roman format",
    )
    args = parser.parse_args()

    with open(args.inpt_bed, "r") as f, open(args.output_bed, "w") as out_f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.strip().split("\t")
            if len(cols) < 1:
                continue
            chrom = cols[0].replace("chr", "")
            # Convert Roman numeral to integer
            roman_chrom = returnRoman(int(chrom))

            out_f.write(f"chr{roman_chrom}\t" + "\t".join(cols[1:]) + "\n")
