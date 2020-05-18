import itertools
import collections
import logging
import os
import time
import sys

"""
Get permuted strepsiptera test frequencies
"""

# set log level
logging.basicConfig(level=logging.INFO)

# create all possible 256 DNA site patterns and save it in a dict
patterns = {}
header = []
for x in itertools.product(*(['ACGT'] * 4)):
    p = ''.join(x)
    patterns[p] = 0
    header.append(p)

logging.debug(patterns)
logging.debug(header)

# specify length of alignments
seq_len=831

# specify in which folder the quartet alignments are
folder='../raw/strepsiptera/quartet-alignments'

# create data file
data_file_name = sys.argv[1] if len(sys.argv) > 1 else '../processed/quartet/strepsiptera/flytrap_freqs_permuted.csv'
data_file = open(data_file_name, 'w')
data_file.write("file," + ','.join(header) + ",label\n")
data_file.flush()
data_file.close()


# compute relative site pattern frequencies for a given MSA
def compute_freq(seq):
    # get and sort all sites
    site_patterns = [''.join(k) for k in zip(*seq.split())]
    site_patterns.sort()
    logging.debug(site_patterns)

    # count how often each site pattern occurs
    abs_frequencies = dict(collections.Counter(site_patterns))
    logging.debug(abs_frequencies)

    # update empty pattern dict with above gathered counts
    patterns_copy=patterns.copy()
    patterns_copy.update(abs_frequencies)

    # compute reltaive frequency for each site pattern
    rel_freq = {k: (v / seq_len) for k, v in patterns_copy.items()}
    return rel_freq


def main():
    # for each quartet number import alignment
    for quartet in range(1,25):
        for filename in os.listdir(folder+'.'):
            if filename[:10]=='quartet{0:03}'.format(quartet):
                with open(folder+filename,'r') as file:
                    msa=file.readlines()

                # extract the single sequences of the alignment
                single_seq=msa[1][11:]+msa[2][11:]+msa[3][11:]+msa[4][11:-1]

                # generate all possible orders of the 4 sequences
                permutations = ['\n'.join(p) for p in itertools.permutations(list(filter(None, single_seq.splitlines())))]

                # gather the site pattern frequencies for the differently ordered alignments
                rel_frequencies = [compute_freq(x) for x in permutations]

                # save site-pattern frequencies in a csv-file
                rows = ["{},".format(filename) + ','.join(map(str, single_freq.values())) + ",{}\n".format(quartet) for single_freq in rel_frequencies]
                data_file = open(data_file_name, 'a')
                data_file.writelines(row for row in rows)

                data_file.flush()
                data_file.close()

    logging.info("Wrote to " + data_file_name + ". Terminating script.")


main()
