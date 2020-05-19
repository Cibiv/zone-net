#!/bin/bash

# simulate data similar to strepsiptera data
python3 data_simulator_strepsiptera.py "../raw/strepsiptera/quartet-tree-parameters/" "../raw/strepsiptera/model-params-msa/param-table.tsv" 20000 '../interim/strepsiptera/sim_freq_train.csv'

# shuffle data
python3 shuffle_data.py -i '../interim/strepsiptera/sim_freq_train.csv' -o '../processed/strepsiptera/train/sim_freq_train_shuffled.csv'
