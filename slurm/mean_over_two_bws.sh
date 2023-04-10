#!/usr/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=200GB
#SBATCH --time=24:00:00

set -e

. "/storage/home/jmy5455/work/miniconda3/etc/profile.d/conda.sh"

conda activate bioinfo

while getopts "hp:g:" opt; do
    case "$opt" in
        h|\?)
            echo "Usage: $(basename $0) [-p] PREFIX [-g] GENOME_FAI FILE1 FILE2"
            exit 1
            ;;
        p) PREFIX=$OPTARG
            ;;
        g) GENOME=$OPTARG
            ;;
    esac
done
shift "$(($OPTIND -1))"

file1=$1
file2=$2

# script under this line
# take mean
wiggletools write_bg ${PREFIX}.bg mean $file1 $file2 
# sort
parsort -k1,1 -k2,2n ${PREFIX}.bg > ${PREFIX}.sort.bg 
# convert bedgraph to bigwig
bedGraphToBigWig ${PREFIX}.sort.bg ~/group/lab/jianyu/genome/mm10/mm10.fa.fai ${PREFIX}.bw 
