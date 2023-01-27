
# load the trained models
import argparse
import logging
import pyBigWig
import torch
import numpy as np
import os, sys, re
sys.path.append("./Bichrom-Regression/trainNN/")
from train import ConvRepeatVGGLike
from data_module import SeqChromDataset, DNA2OneHot, worker_init_fn
from typing import overload, Tuple
from copy import deepcopy
from torch.utils.data import IterableDataset, DataLoader
from glob import glob
from yaml import safe_load
from Bio import SeqIO
from pyfasta import Fasta
from deeplift.dinuc_shuffle import dinuc_shuffle

from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel
)

# Set logger
log_console_format = "[{name:<}] {asctime:<s}: {levelname:<8s} {message:<s}"

main_logger = logging.getLogger("Feature Attribution")
main_logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(log_console_format, style="{"))

main_logger.addHandler(console_handler)

class SeqChromDatasetForBaselines(IterableDataset):
    def __init__(self, fasta, num_chroms=1, seq_transform=DNA2OneHot()):
        super().__init__()
        self.fasta = fasta
        self.fasta_handler = SeqIO.parse(fasta, 'fasta')
        self.fasta_length = len(Fasta(fasta))
        self.num_chroms = num_chroms
        self.seq_transform = seq_transform

    def initialize(self):
        pass
    
    def __len__(self):
        return self.fasta_length
    
    def __iter__(self):
        for record in self.fasta_handler:
            seq = record.seq
            length = len(seq)
            seq = self.seq_transform(seq)
            
            ms = np.vstack([np.zeros(length, dtype=np.float32)]*self.num_chroms)
            
            yield seq, ms
            
# write the attributions into bigwig files
def write_bw(filename, arrs, regions):
    """
    shape of arrs should be (# samples, length)
    Note: Since clustered regions are tiling regions, only base pairs not covered by previous regions would be assigned current region's attribution score
    """
    bw = pyBigWig.open(filename, 'w')
    heads = []
    with open('data/genome/mm10.info', 'r') as f:
        for line in f:
            chrom, length = line.strip().split("\t")
            heads.append((chrom, int(length)))
    heads = sorted(heads, key=lambda x: x[0])
    bw.addHeader(heads)
    lastChrom = -1
    lastEnd = -1
    for arr, region in zip(arrs, regions):
        rchrom, rstart, rend = re.split(r'[:-]', region)
        rstart = int(rstart); rend = int(rend)
        # get uncovered interval (defined by start coordinate `start` and relative start coordinate `start_idx`)
        if lastEnd > rstart and rchrom == lastChrom:
            start_idx = lastEnd - rstart
            start = lastEnd
        else:
            start_idx = 0
            start = rstart
        try:
            bw.addEntries(rchrom, 
                      np.arange(start, rend, dtype=np.int64), 
                      values=arr.astype(np.float64)[start_idx:],
                      span=1)
        except:
            print(rchrom)
            print(start)
            print(rstart)
            print(rend)
            print(lastEnd)
            print(start_idx)
            raise Exception("Runtime error when adding entries to bigwig file, see above region info")
        lastChrom = rchrom
        lastEnd = rend
    bw.close()

def dinuc_shuffle_several_times_seqchrom(seqchrom: Tuple[torch.tensor, torch.tensor], times: int=10, seed: int=1) -> Tuple[torch.tensor, torch.tensor]:
    seq, chrom = seqchrom
    seq_to_return = dinuc_shuffle_several_times_seq(seq)
    chrom_to_return = torch.zeros_like(chrom)    
    return (seq_to_return, chrom_to_return)

def dinuc_shuffle_several_times_seq(seq: torch.tensor, times: int=10, seed: int=1) -> torch.tensor:
    assert len(seq.shape) == 3  # dim: 1 x D x L
    onehot_seq = seq[0].T # dim: L X D
    assert onehot_seq.shape[-1] == 4
    rng = np.random.RandomState(seed)
    device = onehot_seq.device
    to_return = torch.tensor(np.array([dinuc_shuffle(onehot_seq.detach().cpu().numpy(), rng=rng).T for i in range(times)])).to(device)
    return to_return

def hypothetical_attribution_func(multipliers, inputs, baselines):
    """
    Attribution function to compute hypothetical contribution scores
    as inputs for TF-Modisco
    """

    if len(multipliers) == 1:
        mult_seq = multipliers
        input_seq = inputs
        bg_seq = baselines
    elif len(multipliers) == 2:
        mult_seq, mult_chrom = multipliers
        input_seq, input_chrom = inputs
        bg_seq, bg_chrom = baselines
    else:
        raise Exception(f"This script is only compatible with seq or (seq, chrom) input!")
    assert input_seq.shape[-2] == 4
    
    projected_hypothetical_contribs_seq = torch.zeros_like(input_seq)
    for i in range(input_seq.shape[-2]):
        hypothetical_input = torch.zeros_like(input_seq)
        hypothetical_input[:, i, :] = 1.0
        hypothetical_diff_from_bg = hypothetical_input - bg_seq
        hypothetical_contribs = hypothetical_diff_from_bg * mult_seq
        projected_hypothetical_contribs_seq[:, i, :] = torch.sum(hypothetical_contribs.sum(dim=-2))
    
    if len(multipliers) == 1:
        return projected_hypothetical_contribs_seq
    else:
        return projected_hypothetical_contribs_seq, (input_chrom-bg_chrom)*mult_chrom

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_id', help="Model directory name in logger_50epochs_balanced_valtest")
    parser.add_argument('data_config', help="Dataset config yaml file")
    parser.add_argument('cuda_id', help="GPU id", type=int)
    parser.add_argument('input_bed', help="Bed file containing regions to be explained")
    parser.add_argument('out_dir', help="Output directory storing attributions")
    parser.add_argument('--seqonly', action="store_true", default=False, help="If model is seqonly model")
    parser.add_argument('--hypothetical', action="store_true", default=False, help="Compute hypothetical contributions instead, bigwig outputs will be suppressed")
    args = parser.parse_args()
    
    # get cuda device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_id-1)
    device=f'cuda:{args.cuda_id-1}'
    main_logger.info(f'Currently on cuda device: {args.cuda_id-1}')
    # get data config
    config = safe_load(open(args.data_config))
    chrom_name_list = [os.path.basename(i).split('.')[0].replace('-group1', '') for i in config['params']['chromatin_tracks']]
    main_logger.info(f"Chromatin tracks are {chrom_name_list}")

    # load model and attribution method
    config_file = glob(f'./logger_50epochs_balanced_valtest/{args.model_id}/train/version_0/config.yaml')[0]
    checkpoint_file = glob(f'./logger_50epochs_balanced_valtest/{args.model_id}/train/version_0/checkpoints/checkpoint*ckpt')[0]
    logging.info(f'config file is {config_file}\ncheckpoint file is {checkpoint_file}')
    model = ConvRepeatVGGLike.load_from_checkpoint(checkpoint_file)
    model.eval()
    dl = DeepLiftShap(model.to(device))

    # compute attribution scores
    data_ds = SeqChromDataset(args.input_bed, config=safe_load(open(args.data_config)),include_idx=True)
    data_dl = DataLoader(data_ds, batch_size=1, num_workers=4, worker_init_fn=worker_init_fn)
    main_logger.info("Computing feature attributions...")
    seqs = []; chroms = []; attributions = []; regions = []
    for region, seq, chrom in data_dl:
        seqs.append(deepcopy(seq))
        chroms.append(deepcopy(chrom))
        if args.seqonly:
            attribution = dl.attribute(seq.to(device), baselines=dinuc_shuffle_several_times_seq, additional_forward_args=[0], custom_attribution_func=hypothetical_attribution_func if args.hypothetical else None)
        else:
            attribution = dl.attribute((seq.to(device), chrom.to(device)), baselines=dinuc_shuffle_several_times_seqchrom, custom_attribution_func=hypothetical_attribution_func if args.hypothetical else None)
        attribution = attribution.detach().cpu() if args.seqonly else [i.detach().cpu() for i in attribution]
        attributions.append(attribution)
        regions += list(region)
        with torch.no_grad():
            del seq
            del chrom
            del attribution
            torch.cuda.empty_cache()
    main_logger.info("Done!")

    # write output
    # create output directory if not exists
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    # save one-hot coded input and contributions
    main_logger.info("Saving input tensors and corresponding contributions...")
    seqs = torch.cat(seqs)
    attr_seqs = torch.cat(attributions) if args.seqonly else torch.cat([i[0] for i in attributions])
    np.savez(f'{args.out_dir}/{args.model_id}_seq_one_hot.npz', seqs.numpy())
    np.savez(f'{args.out_dir}/{args.model_id}_seq_contribs.npz', attr_seqs.numpy())
    if not args.seqonly:
        chroms = torch.cat(chroms)
        attr_chroms = torch.cat([i[1] for i in attributions])
        np.savez(f'{args.out_dir}/{args.model_id}_chrom.npz', chroms.numpy())
        np.savez(f'{args.out_dir}/{args.model_id}_chrom_contribs.npz', attr_chroms.numpy())
            
    # concatenate attribution tensors, then write to bigwig file
    if not args.hypothetical:
        main_logger.info("Writing attributions into bigwig file...")
        write_bw(f'f{args.out_dir}/{args.model_id}_seq_deepliftSHAP.bw', attr_seqs.sum(dim=1).numpy(), regions)
        main_logger.info(f"Sequence attributions saved intof{args.out_dir}/{args.model_id}_seq_deepliftSHAP.bw")
        if not args.seqonly:
            for idx, c in enumerate(chrom_name_list):
                write_bw(f'f{args.out_dir}/{args.model_id}_{c}_deepliftSHAP.bw', attr_chroms[:,idx,:].numpy(), regions)
                main_logger.info(f"Chromatin feature {c} attributions saved into f{args.out_dir}/{args.model_id}_{c}_deepliftSHAP.bw")
        main_logger.info("All done")
    
