
import os
import subprocess
import argparse
import tensorflow as tf
import numpy as np
import shap
import copy
from functools import partial
from Bio import SeqIO
from deeplift.dinuc_shuffle import dinuc_shuffle
from Bichrom.construct_data import utils

HELP="""

    Run TF-modisco on given fasta sequence file and motif file in meme format

"""

DINUCLEOTIDE_SHUFFLE_PATH="fasta-dinucleotide-shuffle"
MODISCO_PATH="/home/jmy5455/group/lab/jianyu/miniconda3/envs/tfmodisco/bin/modisco"

def load_fa_into_onehot(file):
    """
    Given fasta filename, load into one hot coded numpy array
    """
    seqIO_obj = SeqIO.parse(file, 'fasta')

    seq_onehots = []
    len_expect = -1
    for record in seqIO_obj:
        if len_expect > 0:
            if len(record.seq) != len_expect:
                print(f"Sequence Record {record.id} is of length different from the first record, Skipped")
                continue
        else:
            len_expect = len(record.seq)
        seq_onehot = utils.dna2onehot(record.seq.upper())
        seq_onehots.append(seq_onehot)
    seq_onehots = np.stack(seq_onehots)
    
    return seq_onehots

# function copied from https://github.com/kundajelab/shap/blob/master/notebooks/deep_explainer/Tensorflow%20DeepExplainer%20Genomics%20Example%20With%20Hypothetical%20Importance%20Scores.ipynb
def dinuc_shuffle_several_times(example: list, times: int=10, seed: int=1):
    assert len(example) == 1
    onehot_seq = example[0]
    rng = np.random.RandomState(seed)
    to_return = np.array([dinuc_shuffle(onehot_seq, rng=rng) for i in range(times)])
    return [to_return]

# function copied from https://github.com/kundajelab/shap/blob/master/notebooks/deep_explainer/Tensorflow%20DeepExplainer%20Genomics%20Example%20With%20Hypothetical%20Importance%20Scores.ipynb
def combine_mult_and_diffref_hypo(mult, orig_inp, bg_data):
    to_return = []
    for l in range(len(mult)):
        projected_hypothetical_contribs = np.zeros_like(bg_data[l]).astype("float")
        assert len(orig_inp[l].shape)==2
        #At each position in the input sequence, we iterate over the one-hot encoding
        # possibilities (eg: for genomic sequence, this is ACGT i.e.
        # 1000, 0100, 0010 and 0001) and compute the hypothetical 
        # difference-from-reference in each case. We then multiply the hypothetical
        # differences-from-reference with the multipliers to get the hypothetical contributions.
        #For each of the one-hot encoding possibilities,
        # the hypothetical contributions are then summed across the ACGT axis to estimate
        # the total hypothetical contribution of each position. This per-position hypothetical
        # contribution is then assigned ("projected") onto whichever base was present in the
        # hypothetical sequence.
        #The reason this is a fast estimate of what the importance scores *would* look
        # like if different bases were present in the underlying sequence is that
        # the multipliers are computed once using the original sequence, and are not
        # computed again for each hypothetical sequence.
        for i in range(orig_inp[l].shape[-1]):
            hypothetical_input = np.zeros_like(orig_inp[l]).astype("float")
            hypothetical_input[:,i] = 1.0
            hypothetical_difference_from_reference = (hypothetical_input[None,:,:]-bg_data[l])
            hypothetical_contribs = hypothetical_difference_from_reference*mult[l]
            projected_hypothetical_contribs[:,:,i] = np.sum(hypothetical_contribs,axis=-1)
        to_return.append(np.mean(projected_hypothetical_contribs, axis=0))
    
    return to_return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = HELP,
    )
    parser.add_argument('input_fa', help='Input fasta')
    parser.add_argument('model',help='Model saved in hdf8 format')
    parser.add_argument('motif', help='Motifs saved in MEME format')
    parser.add_argument('prefix', help='Prefix out output')
    parser.add_argument('--window', help='Window size argument passed to modisco motifs', type=int, default=240)
    parser.add_argument('--seed', help='Random seed for generating dinucleotide shuffled background', type=int, default=1)
    parser.add_argument('--background', help='Source of background, mutually exclusive with --background_fa', choices=['dinucleotide_shuffle', 'zero', None], default=None)
    parser.add_argument('--background_fa', help='Fasta file as background, mutually exclusive with --background')
    args = parser.parse_args()
    
    # check background related arguments
    if not args.background and not args.background_fa:
        print(HELP)
        raise Exception("Both --background and --background_fa were supplied, don't know which one to use")
    elif args.background and args.background_fa:
        print(HELP)
        raise Exception("No background related arguments provided, please specify at least one of them")
    
    # get background
    if args.background == 'dinucleotide_shuffle':
        print("Using dinucleotide shuffle function as background...")
        # here background is a function to dinucleotide-shuffle the input examples
        background = partial(dinuc_shuffle_several_times, times=10, seed=args.seed)
    elif args.background == 'zero':
        print("Using zero array as background...")
        background = np.zeros((1, 240, 4))
    else:
        print("Using provided fasta as background...")
        background = load_fa_into_onehot(args.background_fa)
        
    # load original fasta sequence as well
    inputs = load_fa_into_onehot(args.input_fa)

    # load model, get rid of sigmoid activation in the final layer
    tf.compat.v1.disable_v2_behavior()
    model_seq = tf.keras.models.load_model(args.model)
    l = tf.keras.layers.Dense(1, name='dense_linear')
    pre_activation_model = tf.keras.Sequential(model_seq.layers[:-1] + [l])
    l.set_weights(model_seq.layers[-1].get_weights())
    
    # build deepSHAP explainer
    e = shap.DeepExplainer(pre_activation_model, background, combine_mult_and_diffref=combine_mult_and_diffref_hypo)
    shap_values = []
    for i in np.array_split(inputs, 10):
        shap_value = e.shap_values(i, check_additivity=False)[0]    # disable additivity check cuz hypothetical contribs don't sum up to the difference of output
        shap_values.append(shap_value)
    shap_values = np.concatenate(shap_values)
    
    # save shap values and one hot coding array
    if not os.path.isdir("tmp"):
        os.mkdir("tmp")
    shap_values_npz = f"tmp/{args.prefix}_shap_values.npz"
    one_hot_npz = f"tmp/{args.prefix}_one_hot.npz"
    np.savez(shap_values_npz, np.transpose(shap_values, axes=[0,2,1]))
    np.savez(one_hot_npz, np.transpose(inputs, axes=[0,2,1]))
    
    # run tf-modisco
    modisco_results_h5 = f'tmp/{args.prefix}_modisco_results.h5'
    callback = subprocess.run([MODISCO_PATH, 'motifs', '-s', one_hot_npz, '-a', shap_values_npz, '-n', '50000', '-w', str(args.window), '-o', modisco_results_h5], capture_output=True)
    callback.check_returncode()
    print(callback.stdout)
    print(callback.stderr)
    callback = subprocess.run([MODISCO_PATH, 'report', '-i', modisco_results_h5, '-o', f'modisco_reports/{args.prefix}_modisco_report/', '-m', args.motif], capture_output=True)
    callback.check_returncode()
    print(callback.stdout)
    print(callback.stderr)
    
    # remove temporary files
    #os.remove(shap_values_npz)
    #os.remove(one_hot_npz)
    #os.remove(modisco_results_h5)

