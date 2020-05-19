# import neural network as module
from mlp import MLP

# general libraries

import argparse
import numpy as np

import os
import pandas as pd

import matplotlib

matplotlib.use('Agg')

from pylab import *
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", default="config/config_F-zoneNN.yaml", help="config file of net")
ap.add_argument("-n", "--net", default="F-zoneNN_2", help="network-name")
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
directory = "./data/processed/zone/test/"+args['seqLen']+"bp/"

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

df_output=pd.merge(df_farris, df_fels, on=['len','pProb','qProb'], how='outer')
df_output=df_output.rename(index=str, columns={"accuracy_x": "Farris", "accuracy_y": "Felsenstein"}).drop(['len'],axis=1)
df_output['Farris']=(df_output['Farris']/100).round(2)
df_output['Felsenstein']=(df_output['Felsenstein']/100).round(2)
df_output=df_output[['pProb','qProb','Farris','Felsenstein']]

df_output.to_csv(args["output"]+'/test_'+net+ '_seqLen_'+ args['seqLen'] +'.csv', index=False)

def testToAccuracy(df):
    df['probp']=df['pProb']
    df['probq']=df['qProb']
    far_df=df
    far_df['acc']=far_df['Farris']
    far_df=far_df.drop(['pProb','qProb','Felsenstein','Farris'], axis=1)
    fel_df=df
    fel_df['acc']=fel_df['Felsenstein']
    fel_df=fel_df.drop(['pProb','qProb','Felsenstein','Farris'], axis=1)

    avg_df=df
    avg_df['acc']=(avg_df['Felsenstein']+avg_df['Farris'])*0.5
    avg_df=avg_df.drop(['pProb','qProb','Felsenstein','Farris'], axis=1)

    return far_df, fel_df, avg_df

far_nn, fel_nn, avg_nn = testToAccuracy(df_output)

print(far_nn['acc'].mean())
print(fel_nn['acc'].mean())
print(avg_nn['acc'].mean())

def getPivot(file):
    df=file
    df=df.sort_values(['probp', 'probq'], ascending=[True, True])
    df['acc']=df['acc'].round(2)
    df=df.pivot("probp", "probq", "acc")

    return df

far_nn=getPivot(far_nn)
fel_nn=getPivot(fel_nn)

def make_figure(far, fel, name):
    fig = plt.figure(figsize=(18,7.5))
    cbar_ax = fig.add_axes([.125, 0.02, .775, .05])

    subplot(1,2,1)
    ax=sns.heatmap(far, annot=False, cmap="YlOrRd_r", square=True, cbar=True, center=0.5, vmin=0, vmax=1, cbar_ax=cbar_ax, cbar_kws={"orientation": "horizontal"})
    ax.text((ax.get_xlim()[1])/2, ax.get_ylim()[0]+0.5, "Farris", fontsize=22, horizontalalignment='center')
    ax.invert_yaxis()
    ax.set_xlabel("q (3 branches)", fontsize=24)
    ax.set_ylabel("p (2 branches)", fontsize=24)
    plt.yticks(np.arange(0,15,2),("0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7"), rotation=0, fontsize="16")
    plt.xticks(np.arange(0,15,2),("0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7"), rotation=0, fontsize="16")

    subplot(1,2,2)
    ax=sns.heatmap(fel, annot=False, cmap="YlOrRd_r", square=True, cbar=False, center=0.5, vmin=0, vmax=1)
    ax.text((ax.get_xlim()[1])/2, ax.get_ylim()[0]+0.5, "Felsenstein", fontsize=22, horizontalalignment='center')
    ax.invert_yaxis()
    ax.set_xlabel("q (3 branches)", fontsize=24)
    ax.set_ylabel("p (2 branches)", fontsize=24)
    plt.yticks(np.arange(0,15,2),("0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7"), rotation=0, fontsize="16")
    plt.xticks(np.arange(0,15,2),("0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7"), rotation=0, fontsize="16")
    cbar_ax.set_xticklabels(['0%','20%','40%','60%','80%','100%'], fontsize=24)
    subplots_adjust(left  = 0.125, right = 0.9, bottom = 0.25, top = 0.9, wspace = 0.4, hspace = 0.2)

    fig.savefig(name, bbox_inches='tight', dpi=100)

make_figure(far_nn, fel_nn, args["output"]+'/heatmap_permuted_dataset_' + net + '_seqLen_' + args['seqLen'])
