#!/bin/bash

# permute test data
python3 permute_data_sim_jc.py -i ../raw/sim_jc/test/test_sim_jc_all_1000bp.csv -o ../interim/sim_jc/test/test_sim_jc_all_1000bp_permuted.csv

# split test data into data files which contain only data of one p-q-combination
python3 split_test_data_sim_jc.py -i test_sim_jc_all_1000bp_permuted.csv -o ../processed/sim_jc/test/1000bp
