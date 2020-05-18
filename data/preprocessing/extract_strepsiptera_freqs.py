import itertools
import collections
import logging
import os
import time

"""
Extracts site pattern frequencies from Strepsiptera multiple sequence alignments.
"""

# set log level
logging.basicConfig(level=logging.INFO)

# create all possible 256 DNA site patterns
patterns = {}

header = []

for x in itertools.product(*(['ACGT'] * 4)):
    p = ''.join(x)
    patterns[p] = 0
    header.append(p)

logging.debug(patterns)
logging.debug(header)

# create data file
data_file_name = sys.argv[1] if len(sys.argv) > 1 else '../processed/quartet/strepsiptera/flytrap_freqs.csv'
data_file = open(data_file_name, 'w')
data_file.write("file," + ','.join(header) + ",label\n")
data_file.flush()
data_file.close()

# sequence params
seq_len = 831

# compute relative site pattern frequencies
def compute_freq(seq):
    site_patterns = [''.join(k) for k in zip(*seq.split())]
    site_patterns.sort()
    site_patterns = [x.upper() for x in site_patterns]
    logging.debug(site_patterns)
    print(site_patterns)

    abs_frequencies = dict(collections.Counter(site_patterns))
    logging.debug(abs_frequencies)

    patterns_c=patterns.copy()
    patterns_c.update(abs_frequencies)

    rel_freq = {k: (v / seq_len) for k, v in patterns_c.items()}
    return rel_freq


data_file = open(data_file_name, 'a')


path = '../raw/strepsiptera/quartet-alignments/'

# loop through .phy files
for subdir, dirs, files in os.walk(path):
    for file in files:
        msa = open(path + file).readlines()
        logging.debug(msa)

        single_seq = '\n'.join([x[10:] for x in msa[1:5]])
        logging.debug(single_seq)

        rel_frequencies = [compute_freq(single_seq)]
        logging.debug(rel_frequencies)

        data_file.writelines([file + "," + ','.join(map(str, single_freq.values())) + "," + '1000' + "\n" for single_freq in rel_frequencies])

        data_file.flush()


data_file.close()

logging.info("Wrote to " + data_file_name + ". Terminating script.")
