#!/usr/bin/env python3

DESCRIPTION = """
  Swap the motif name and id in a meme file for modisco report
"""

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "inpt_meme",
        type=str,
        help="Input MEME file with motif names and IDs",
    )
    parser.add_argument(
        "output_meme",
        type=str,
        help="Output MEME file with swapped motif names and IDs",
    )
    args = parser.parse_args()

    with open(args.inpt_meme, "r") as f, open(args.output_meme, "w") as out_f:
        for line in f:
            if line.startswith("MOTIF"):
                parts = line.strip().split()
                if len(parts) >= 3:
                    # Swap the motif name and ID
                    parts[1], parts[2] = parts[2], parts[1]
                    out_f.write(" ".join(parts) + "\n")
                else:
                    out_f.write(line)
            else:
                out_f.write(line)
