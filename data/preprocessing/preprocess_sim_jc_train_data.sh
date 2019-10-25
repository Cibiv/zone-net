#!/bin/bash

# permute training data
python3 permute_data_sim_jc.py -i ../raw/sim_jc/train/train_sim_jc.csv -o ../interim/sim_jc/train/train_sim_jc_permuted.csv

# shuffle training data
python3 shuffle_data.py -i ../interim/sim_jc/train/train_sim_jc_permuted.csv -o ../processed/sim_jc/train/train_sim_jc_permuted_shuffled.csv
