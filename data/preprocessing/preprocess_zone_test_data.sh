#!/bin/bash

# permute test data
python3 permute_data_zone.py -i ../raw/zone/test/test_zone_all_1000bp.csv -o ../interim/zone/test/test_zone_all_1000bp_permuted.csv

# split test data into data files which contain only data of one p-q-combination
python3 split_test_data_zone.py -i test_zone_all_1000bp_permuted.csv -o ../processed/zone/test/1000bp
