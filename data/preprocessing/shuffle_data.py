import random
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='path of input file')
parser.add_argument('-o', '--output', required=True, help='path of output file')
parser.add_argument('-s', '--seed', default=100, type=int, help='seed used for shuffling')
args = vars(parser.parse_args())

# extract header of input file and exclude from data
file = open(args['input'], 'r', encoding='utf8')
rows = file.readlines()
header = rows[0]
rows = rows[1:]
file.close()

# shuffle data using a seed such that the result can be reproduced
random.seed(args['seed'])
random.shuffle(rows)

file = open(args['output'], 'w')
# write header to output file
file.writelines(header)
# write shuffled data to output file
file.writelines(rows)
file.close()

logging.info("Wrote shuffled data to " + args['output'] + ".")
