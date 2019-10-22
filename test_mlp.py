# import neural network as module
from mlp import MLP

# general libraries
import argparse
import os
# import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", default="config/config_net_sim_jc.yaml", help="config file of net")
ap.add_argument("-n", "--net", default="net_sim_jc_2", help="network-name")
ap.add_argument("-o", "--output", default=".", help="output directory for heatmap")
ap.add_argument("-l", "--seqLen", default="1000", help="sequence length for testing")

args = vars(ap.parse_args())

# initialize network instance
nn = MLP(args["config"])

# define network parameters
nn.test_only = True
nn.multiple_data = False
net = args["net"]
save = True
nn.read_network_from = 'models/' + net

# initialize directory containing test data
directory = "./data/processed/sim_jc/test/"+args['seqLen']+"bp/"

tmp_accuracies = {'pProb': [], 'qProb': [], 'len': [], 'zone': [], 'accuracy': []}

# loop through test files and evaluate network performance
for subdir, dirs, files in os.walk(directory):
    for file in files:
        if os.fsdecode(file).endswith(".csv"):
            file = os.path.join(subdir, file)
            nn.data_file = file
            accuracies = nn.test()
            for key in accuracies:
                if key == 'felsenstein' or key == 'farris':
                    tmp_accuracies['len'].append(args['seqLen'])
                    tmp_accuracies['pProb'].append(float(file[file.rfind("_p")+2:file.rfind("_q")]))
                    tmp_accuracies['qProb'].append(float(file[file.rfind("_q")+2:file.rfind(".csv")]))
                    tmp_accuracies['zone'].append(key)
                    tmp_accuracies['accuracy'].append(accuracies[key])
    break

# transform dict into pandas data frame
df = pd.DataFrame.from_dict(tmp_accuracies)

# separate data into zones
df_fels = df.loc[df['zone'] == 'felsenstein'].drop('zone', axis=1)
df_farris = df.loc[df['zone'] == 'farris'].drop('zone', axis=1)

df_fels = df_fels.sort_values(['pProb', 'len'], ascending=[True, True])
df_farris = df_farris.sort_values(['pProb', 'len'], ascending=[True, True])

# df_output = pd.merge(df_farris, df_fels, on=['len','pProb','qProb'], how='outer')
# df_output = df_output.rename(index=str, columns={"accuracy_x": "Far", "accuracy_y": "Fel"}).drop(['len'],axis=1)
# df_output['Prob'] = 'p'+df_output['pProb'].apply(lambda x: round(x, 3)).astype(str)+'_q'+df_output['qProb'].apply(lambda x: round(x, 3)).astype(str)
# df_output = df_output[['Prob','Far','Fel']]

# df_output.to_csv(args["output"]+'test_'+net+ '_seqLen_'+ args['seqLen'] +'.csv', index=False)

# TODO: logging
print('accuracy for farris: %f' %(df_farris['accuracy'].mean()))
print('accuracy for felsenstein: %f' %(df_fels['accuracy'].mean()))
print('overall accuracy: %f' %((df_farris['accuracy'].mean() + df_fels['accuracy'].mean())/2))

# convert to matrix for heatmap
far_matrix = df_farris.pivot("pProb", "qProb", "accuracy")
fel_matrix = df_fels.pivot("pProb", "qProb", "accuracy")


fig_far = plt.figure(figsize=(12, 12))
sns.set(font_scale=1.5)
r = sns.heatmap(far_matrix, cmap='RdYlGn', center=50, vmin=0, vmax=100, annot=True, annot_kws={'size': 16}, fmt='.0f')
r.set_title("Heatmap of accuracies for Farris (for different p and q values)", fontsize=20)
xlabels = ['%.3f' % float(t.get_text()) for t in r.get_xticklabels()]
r.set_xticklabels(xlabels)
ylabels = ['%.3f' % float(t.get_text()) for t in r.get_yticklabels()]
r.set_yticklabels(ylabels)
r.invert_yaxis()
plt.savefig(args["output"]+'/heatmap_far_permuted_dataset_' + net + '_seqLen_' + args['seqLen'])


fig_fel = plt.figure(figsize=(12, 12))
sns.set(font_scale=1.5)
r = sns.heatmap(fel_matrix, cmap='RdYlGn', center=50, vmin=0, vmax=100, annot=True, annot_kws={'size': 16}, fmt='.0f')
r.set_title("Heatmap of accuracies for Felsenstein (for different p and q values)", fontsize=20)
xlabels = ['%.3f' % float(t.get_text()) for t in r.get_xticklabels()]
r.set_xticklabels(xlabels)
ylabels = ['%.3f' % float(t.get_text()) for t in r.get_yticklabels()]
r.set_yticklabels(ylabels)
r.invert_yaxis()
plt.savefig(args["output"]+'/heatmap_fels_permuted_dataset_' + net + '_seqLen_' + args['seqLen'])
