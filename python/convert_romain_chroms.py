#!/usr/bin/env python3

DESCRIPTION = """
Convert Roman chromosome names to integer, assume the first column is the chromosome name in format "chrI", "chrII", ..., "chrX", etc. Note chrX is treated as chr10, there is no X, Y, M chromosome assumed
"""

from argparse import ArgumentParser


class Solution(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        roman = {
            "I": 1,
            "V": 5,
            "X": 10,
            "L": 50,
            "C": 100,
            "D": 500,
            "M": 1000,
            "IV": 4,
            "IX": 9,
            "XL": 40,
            "XC": 90,
            "CD": 400,
            "CM": 900,
        }
        i = 0
        num = 0
        while i < len(s):
            if i + 1 < len(s) and s[i : i + 2] in roman:
                num += roman[s[i : i + 2]]
                i += 2
            else:
                # print(i)
                num += roman[s[i]]
                i += 1
        return num


if __name__ == "__main__":
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "inpt_bed",
        type=str,
        help="Input BED file with chromosome names in Roman format",
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
            ob = Solution()
            int_chrom = ob.romanToInt(chrom)

            out_f.write(f"chr{int_chrom}\t" + "\t".join(cols[1:]) + "\n")
