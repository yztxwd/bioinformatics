#!python3

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

