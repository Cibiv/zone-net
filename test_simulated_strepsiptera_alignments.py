# import neural network as module
from mlp import MLP

# general libraries
import argparse
import logging

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", default="config/config_StrepsipteraNN.yaml", help="config file of net")
ap.add_argument("-n", "--net", default="StrepsipteraNN_3", help="network-name")
ap.add_argument("-i", "--input", default="./data/processed/strepsiptera/test/sim_freq_test.csv", help="input file for testing")
args = vars(ap.parse_args())


# initialize network instance
nn = MLP(args["config"])

# define network parameters
net = args["net"]
nn.read_network_from = 'models/' + net

# get test file and evaluate network performance
nn.data_file = args['input']
accuracies = nn.test()

output_file="accuracies_{}_on_{}.csv".format(args['net'],args['input'][args['input'].rfind('/')+1:args['input'].rfind('.csv')])

with open(output_file, 'w+') as file:
    file.writelines(["Accuracies of {} on file '{}':\n".format(args['net'],args['input']),
                    "True Farris-type trees: {}%\n".format(round(accuracies['farris'],2)),
                    "True Felsenstein-type trees: {}%\n".format(round(accuracies['felsenstein'],2)),
                    "Average: {}%\n".format(round(accuracies['all'],2))])
    file.close()

logging.info("Wrote accuracies to " + output_file + ".")
