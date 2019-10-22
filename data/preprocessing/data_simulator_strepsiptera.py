import itertools
import collections
import subprocess
import logging
import os
import sys
import time
import multiprocessing

"""
Simulates input for neural network that learns to classify multiple sequence alignments (encoded as
base frequencies) based on four-taxon tree to be either "long branches together" or "long branches separated".

For a given set of substitution models (F81+G', 'F81', 'GTR+G', 'GTR', 'HKY+G', 'HKY', 'JC+G', 'JC', 'K2P+G',
K2P, TN+G, TN) multiple sequence alignments are simulated using seq-gen. 24 four-taxon trees (with different p
and q values based on the flytrap data) and 2 different tree topologies are used for the simulation:

Topologies:
===========

Felsenstein-type tree:  (AB|CD)

   B    D
    \__/
    /  \
   /    \
  /      \
 /        \
A          C

Farris-type tree:       (AC|BD)

   B   D
    \ /
     |
     |
    / \
   /   \
  /     \
 /       \
A         C

The MSAs are then permuted to allow the network to be permutation invariant (4! = 24 permutations per alignment).

Then, the 4^4 = 256 site pattern frequencies are computed for each alignment and the data is stored in a file.

The number of data points can be computed as follows:

    `2 zones x 2 sets of GTR parameters (with and without outgroup) x 12 substitution models x 24 trees
    x 24 permutations x <number-of-iterations> = 27648 x <number-of-iterations>`

The default value for <number-of-iterations> is 10,000, resulting in 276.48 mio data points (site pattern frequencies)

"""


# set log level
logging.basicConfig(level=logging.INFO)

# simulation parameters
model_params_path = sys.argv[2] if len(sys.argv) > 2 else "../raw/strepsiptera/model-params-msa/param-table.tsv"
quartet_trees_path = sys.argv[1] if len(sys.argv) > 1 else "../raw/strepsiptera/quartet-tree-parameters/"
seq_len = 1000
num_it = sys.argv[3] if len(sys.argv) > 3 else 10000

# create all possible 256 DNA site patterns
patterns = {}
models = ['F81+G', 'F81', 'GTR+G', 'GTR', 'HKY+G', 'HKY', 'JC+G', 'JC', 'K2P+G', 'K2P', 'TN+G', 'TN']

header = []

for x in itertools.product(*(['ACGT'] * 4)):
    p = ''.join(x)
    patterns[p] = 0
    header.append(p)

logging.debug(patterns)
logging.debug(header)

# create data file
data_file_name = sys.argv[4] if len(sys.argv) > 4 else '../interim/strepsiptera/sim_freq_' + time.strftime("%Y%m%d_%H%M%S") + '.csv'
data_file = open(data_file_name, 'w')
data_file.write("seed,rAC,rAG,rAT,rCG,rCT,rGT,fA,fC,fG,fT,alpha," + ','.join(header) + ",label\n")
data_file.flush()
data_file.close()


# obtain trees in Newick format per substitution model for MSA simulation
def get_tree_params(path):
    zones = ['fel', 'far']
    # dict with key = model, value = list with trees
    trees = {m: {z: [] for z in zones} for m in models}
    for subdir, dirs, files in os.walk(path):
        for file_name in files:
            tree_file = path + file_name
            for model in models:
                if "trees-quartet-" + model + ".tsv" in file_name:  # if it is the file that contains the given model
                    logging.debug("model: " + model + " in file " + file_name)
                    with open(tree_file, 'r') as f:
                        next(f)  # skip header row
                        for line in f.readlines():
                            for zone, loc in zip(zones, [3, 10]):
                                trees[model][zone].append(line.split()[loc])

    return trees

# obtain GTR parameter combinations (base frequencies, transition/transversion rates and alpha) for MSA simulation
def get_gtr_params(param_table):
    gtr_params = {m: [] for m in models}
    with open(param_table, 'r') as f:
        next(f)  # skip header row
        for line in f.readlines():
            gtr_params[line.split()[1]].append({
                "freqs": ','.join([line.split()[8], line.split()[9], line.split()[10], line.split()[11]]),
                "rates": ','.join([line.split()[2], line.split()[3], line.split()[4],
                                   line.split()[5], line.split()[6], line.split()[7]]),
                "alpha": line.split()[12]})
    return gtr_params


# return all possible combinations of trees (in zones) and GTR parameters with same substitution model
def marry_trees_gtr(tree_params, gtr_params):
    newly_wed = {'model': [], 'gtr': [], 'zone': [], 'tree': []}
    for model in models:
        for zone in tree_params[model].keys():
            for gtr_param_set in gtr_params[model]:
                for tree in tree_params[model][zone]:
                    newly_wed['model'].append(model)
                    newly_wed['gtr'].append(gtr_param_set)
                    newly_wed['zone'].append(zone)
                    newly_wed['tree'].append(tree)
    return newly_wed


# compute relative site pattern frequencies for a given MSA
def compute_freq(seq):
    site_patterns = [''.join(k) for k in zip(*seq.split())]
    site_patterns.sort()
    logging.debug(site_patterns)

    abs_frequencies = dict(collections.Counter(site_patterns))
    logging.debug(abs_frequencies)

    patterns.update(abs_frequencies)

    rel_freq = {k: (v / seq_len) for k, v in patterns.items()}
    return rel_freq


# conduct MSA simulation
def simulate(params):

    label = "0" if params[2] == 'fel' else "1"

    rand_seed = time.time()

    # call seq-gen and generate MSAs for current GTR parameter combination
    alpha = ' -a' + str(params[1]['alpha']) if params[1]['alpha'] != 1000 else ""
    execution_str = 'seq-gen -mGTR -q' + alpha + ' -z' + str(rand_seed) + ' -l ' + str(seq_len) \
                    + ' -f' + str(params[1]['freqs']) + ' -r' + str(params[1]['rates']) \
                    + ' <<< "' + str(params[3]) + '"'
    logging.debug(execution_str)

    proc = subprocess.Popen([execution_str], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            shell=True, executable='/bin/bash')

    msa = [x.decode() for x in proc.stdout.readlines()]
    logging.debug(msa)

    single_seq = '\n'.join([x[10:] for x in msa[1:5]])
    logging.debug(single_seq)

    permutations = ['\n'.join(p) for p in itertools.permutations(list(filter(None, single_seq.splitlines())))]
    logging.debug(permutations)

    rel_frequencies = [compute_freq(x) for x in permutations]
    logging.debug(rel_frequencies)

    rows = [str(rand_seed) + "," + str(params[1]['rates']) + "," + str(params[1]['freqs']) + ","
            + str(params[1]['alpha']) + "," + ','.join(map(str, single_freq.values())) + "," + str(label)
            + "\n" for single_freq in rel_frequencies]

    return rows


def main(tree_path, gtr_path):

    for i in range(0, num_it):

        tree_params = get_tree_params(tree_path)
        gtr_params = get_gtr_params(gtr_path)

        param_dict = marry_trees_gtr(tree_params, gtr_params)
        params = zip(*param_dict.values())

        # create multiprocessing pool with number of processes equal to number of cores
        pool = multiprocessing.Pool()

        logging.info("Starting worker pool.")

        result = pool.map(simulate, params)

        pool.close()

        logging.info("Pool is done. Writing to file.")

        data_file = open(data_file_name, 'a')

        data_file.writelines(row for perm in result for row in perm)

        data_file.flush()

        data_file.close()

    logging.info("Wrote to " + data_file_name + ". Terminating script.")


main(quartet_trees_path, model_params_path)
