import argparse
import pandas as pd
import logging

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='input file in interim folder')
parser.add_argument('-o', '--output', required=True, help='folder for output')
args = vars(parser.parse_args())

# list of the p-, q-values
lengths=[0.025,0.075,0.125,0.175,0.225,0.275,0.325,0.375,0.425,0.475,0.525,0.575,0.625,0.675,0.725]

# read file of all site pattern frequencies
df=pd.read_csv('../interim/zone/test/' + args['input'])

# split input into several data files which contain data for specific p-,q-values
for p in lengths:
    for q in lengths:
        data = df[(df['pProb']==p) & (df['qProb']==q)]
        data.to_csv(args['output'] + '/' + args['input'][:-4] + '_p' + str(p) + '_q' + str(q) + '.csv', index=False)

logging.info("Saved the datafiles to the folder '" + args['output'] + "'.")
