#!/bin/bash

# simulate data similar to strepsiptera data
python3 data_simulator_strepsiptera.py "../raw/strepsiptera/quartet-tree-parameters/" "../raw/strepsiptera/model-params-msa/param-table.tsv" 10 '../processed/strepsiptera/test/sim_freq_test.csv'
