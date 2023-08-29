#!/bin/bash

USAGE="./compute_FRiP.sh sample.narrowPeak sample.bam"

SAF=`basename ${1} .narrowPeak`.saf

# convert narrowPeak to SAF foramt
awk 'BEGIN{FS=OFS="\t"; print "GeneID\tChr\tStart\tEnd\tStrand"}{print $4, $1, $2+1, $3, "."}' ${1} > $SAF

# count fraction of reads falling in peak regions
featureCounts -a ${SAF} -F SAF -o readCountInPeaks.txt ${2}

# remove temporary saf file
rm $SAF
