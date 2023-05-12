
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
from typing import Tuple, Any
from torch import Tensor
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    ExpansionTypes)
from captum._utils.typing import TargetType

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

# overwrite the expand and compute_mean function of DeepLiftShap to enable example specific background sets
class DeepLiftShapPatched(DeepLiftShap):

    def _expand_inputs_baselines_targets(
        self,
        baselines: Tuple[Tensor, ...],
        inputs: Tuple[Tensor, ...],
        target: TargetType,
        additional_forward_args: Any,
        num_baselines_per_sample: int=10,
    ) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...], TargetType, Any]:

        inp_bsz = inputs[0].shape[0]
        #base_bsz = baselines[0].shape[0]
        base_bsz = num_baselines_per_sample

        expanded_inputs = tuple(
            [
                input.repeat_interleave(base_bsz, dim=0).requires_grad_()
                for input in inputs
            ]
        )
        # no need to repeat baselines here, we already did it in the baseline function
        #expanded_baselines = tuple(
        #    [
        #        baseline.repeat(
        #            (inp_bsz,) + tuple([1] * (len(baseline.shape) - 1))
        #        ).requires_grad_()
        #        for baseline in baselines
        #    ]
        #)
        expanded_baselines = baselines
        #print(len(expanded_inputs))
        #print(expanded_inputs[0].shape)
        #print(expanded_baselines[0].shape)
        #print(expanded_baselines[1].shape)
        expanded_target = _expand_target(
            target, base_bsz, expansion_type=ExpansionTypes.repeat_interleave
        )
        input_additional_args = (
            _expand_additional_forward_args(
                additional_forward_args,
                base_bsz,
                expansion_type=ExpansionTypes.repeat_interleave,
            )
            if additional_forward_args is not None
            else None
        )
        return (
            expanded_inputs,
            expanded_baselines,
            expanded_target,
            input_additional_args,
        )

    def _compute_mean_across_baselines(
        self, inp_bsz: int, base_bsz: int, attribution: Tensor, num_baselines_per_sample: int=10
    ) -> Tensor:
        # Average for multiple references
        attr_shape: Tuple = (inp_bsz, num_baselines_per_sample)
        if len(attribution.shape) > 1:
            attr_shape += attribution.shape[1:]
        return torch.mean(attribution.view(attr_shape), dim=1, keepdim=False)
            
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
    seq_to_return = dinuc_shuffle_several_times_seq(seq, times, seed)
    chrom_to_return = torch.stack([torch.zeros_like(i) for j in range(times) for i in chrom])
    return (seq_to_return, chrom_to_return)

def dinuc_shuffle_several_times_seq(seq: torch.tensor, times: int=10, seed: int=1) -> torch.tensor:
    assert len(seq.shape) == 3  # dim: N x D x L
    onehot_seq = torch.permute(seq, (0, 2, 1)) # dim: N x L x D
    assert onehot_seq.shape[-1] == 4
    device = onehot_seq.device
    to_returns = []
    for s in onehot_seq:
        # reset RandomState every loop to make it the same as v2 behavior on each example
        rng = np.random.RandomState(seed)
        to_return = torch.tensor(np.array([dinuc_shuffle(s.detach().cpu().numpy(), rng=rng).T for i in range(times)])).to(device)
        to_returns.append(to_return)
    return torch.cat(to_returns)

def hypothetical_attribution_func(multipliers, inputs, baselines):
    """
    Attribution function to compute hypothetical contribution scores
    as inputs for TF-Modisco
    """

    assert isinstance(inputs, Tuple)

    if len(multipliers) == 1:
        mult_seq = multipliers[0]
        input_seq = inputs[0]
        bg_seq = baselines[0]
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
        projected_hypothetical_contribs_seq[:, i, :] = hypothetical_contribs.sum(dim=-2)
    
    if len(multipliers) == 1:
        return (projected_hypothetical_contribs_seq, )
    else:
        return projected_hypothetical_contribs_seq, (input_chrom-bg_chrom)*mult_chrom

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_id', help="Model directory name in logger_50epochs_balanced_valtest")
    parser.add_argument('data_config', help="Dataset config yaml file")
    parser.add_argument('input_bed', help="Bed file containing regions to be explained")
    parser.add_argument('out_dir', help="Output directory storing attributions")
    parser.add_argument('--seqonly', action="store_true", default=False, help="If model is seqonly model")
    parser.add_argument('--hypothetical', action="store_true", default=False, help="Compute hypothetical contributions instead, bigwig outputs will be suppressed")
    args = parser.parse_args()
    
    # get cuda device
    device=f'cuda:0'
    main_logger.info(f'Using device {device}')
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
    dl = DeepLiftShapPatched(model.to(device))

    # compute attribution scores
    data_ds = SeqChromDataset(args.input_bed, config=safe_load(open(args.data_config)),include_idx=True)
    data_dl = DataLoader(data_ds, batch_size=128, num_workers=16, worker_init_fn=worker_init_fn)
    main_logger.info("Computing feature attributions...")
    seqs = []; chroms = []; attributions = []; regions = []
    for region, seq, chrom in data_dl:
        if args.hypothetical:
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
    # create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    attr_seqs = torch.cat(attributions) if args.seqonly else torch.cat([i[0] for i in attributions])
    attr_chroms = torch.cat([i[1] for i in attributions])
    if args.hypothetical:
        # save one-hot coded input and contributions
        main_logger.info("Saving input tensors and corresponding contributions...")
        seqs = torch.cat(seqs)
        np.savez(f'{args.out_dir}/{args.model_id}_seq_one_hot.npz', seqs.numpy())
        np.savez(f'{args.out_dir}/{args.model_id}_seq_contribs.npz', attr_seqs.numpy())
        if not args.seqonly:
            chroms = torch.cat(chroms)
            np.savez(f'{args.out_dir}/{args.model_id}_chrom.npz', chroms.numpy())
            np.savez(f'{args.out_dir}/{args.model_id}_chrom_contribs.npz', attr_chroms.numpy())
    else:
        # concatenate attribution tensors, then write to bigwig file
        main_logger.info("Writing attributions into bigwig file...")
        write_bw(f'{args.out_dir}/{args.model_id}_seq_deepliftSHAP.bw', attr_seqs.sum(dim=1).numpy(), regions)
        main_logger.info(f"Sequence attributions saved intof{args.out_dir}/bigwigs/{args.model_id}_seq_deepliftSHAP.bw")
        if not args.seqonly:
            for idx, c in enumerate(chrom_name_list):
                write_bw(f'{args.out_dir}/{args.model_id}_{c}_deepliftSHAP.bw', attr_chroms[:,idx,:].numpy(), regions)
                main_logger.info(f"Chromatin feature {c} attributions saved into f{args.out_dir}/bigwigs/{args.model_id}_{c}_deepliftSHAP.bw")
        main_logger.info("All done")
    
