# import neural network as module
from mlp import MLP

# general libraries

import argparse
import numpy as np
import tensorflow as tf
import os
import pandas as pd
import logging

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", default="config/config_StrepsipteraNN.yaml", help="config file of net")
ap.add_argument("-n", "--net", default="StrepsipteraNN_3", help="network-name")
ap.add_argument("-p","--permuted", action="store_true", help="True if all different orderings of taxa for quartets should be used")

args = vars(ap.parse_args())


# initialize network instance
nn = MLP(args["config"])

# define network parameters
net = args["net"]
nn.read_network_from = 'models/' + net

# get test file and evaluate network performance
if args["permuted"]:
    perm="_permuted"
else:
    perm=""

nn.data_file = './data/processed/strepsiptera/quartet/strepsiptera_freqs' + perm + '.csv'

#tf.logging.info("Reading from ", path)
df = pd.read_csv(nn.data_file)
output_df = df.iloc[:, :1]
df = df.iloc[:, 1:]
# separate data into features and labels
x_data = np.array(df.iloc[:, 0:-nn.layers[-1]].values)
y_data = np.array(df.iloc[:, -nn.layers[-1]:].values)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

output = nn.prediction(X, False, False)
saver = tf.train.Saver()
# initialize network
init = tf.global_variables_initializer()
# start session
sess = tf.InteractiveSession()
sess.run(init)
saver.restore(sess, os.path.join('./', nn.read_network_from))
outputs = sess.run(output, feed_dict={X: x_data, Y: y_data})
sess.close()

output_df['output']=outputs
output_df['quartet']=output_df['file']
output_df['Farris']=np.where(output_df['output']>=0.5,1,0)

# collect number of quartets which are predicted to be Farris
pivot = pd.pivot_table(output_df, values='Farris', index=['quartet'], aggfunc=np.sum).reset_index()
# for all other quartets the network infers a Felsenstein-type tree
pivot['Felsenstein']=int(len(output_df)/24)-pivot['Farris']

output_file = 'test_' + net + '_strepsiptera_quartets' +perm + '.csv'

pivot.to_csv(output_file, index=False)

logging.info("Wrote to " + output_file + ".")
