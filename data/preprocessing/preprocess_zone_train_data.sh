#!/bin/bash

# permute training data
python3 permute_data_zone.py -i ../raw/zone/train/train_zone.csv -o ../interim/zone/train/train_zone_permuted.csv

# shuffle training data
python3 shuffle_data.py -i ../interim/zone/train/train_zone_permuted.csv -o ../processed/zone/train/train_zone_permuted_shuffled.csv
