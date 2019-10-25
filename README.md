# Distinguishing Felsenstein zone from Farris zone

## Prerequisites

For all scripts Python 3.6.6 was used. The python packages used can be installed via
```sh
pip install (--user) -r packages_required.txt
```
if python3 and pip are already installed.

To pull the raw data from this repository, please make sure that you have [Git Large File Storage](https://git-lfs.github.com/) installed.

## Network for simulated alignments using Jukes-Cantor model

The results presented in the manuscript stem from the network `net_sim_jc_2` saved in the models folder. This model can be retrained by first preprocessing the training data via 
```sh
./preprocess_sim_jc_train_data.sh
```
in the folder data/preprocessing and then training the network using `config_net_sim_jc.yaml` in the config folder
```sh
python3 mlp.py config/config_net_sim_jc.yaml
```
If an already trained network should be tested, first the test data has to be preprocessed by
```sh
./preprocess_sim_jc_test_data.sh
```
in the folder data/preprocessing.

Testing of the network can be done by executing:
```sh
python3 test_mlp.py [-h]
```

## Network for Strepsiptera data

The results presented in the manuscript stem from the network `net_strepsiptera_4` saved in the models folder. The preprocessing for the training data
```sh
./preprocess_strepsiptera_train_data.sh
```
and for the test data
```sh
./preprocess_strepsiptera_test_data.sh
```
can both be started in the folder data/preprocessing. The same holds for the preprocessing of the Strepsiptera data
```sh
./preprocess_strepsiptera_real_quartets.sh
```
The network can then be trained via
```sh
python3 mlp.py config/config_net_strepsiptera.yaml train
```
<!--and tested on simulated data by
```sh
./test_strepsiptera_sim.sh
```
The network can be tested on the real Strepsiptera quartets by
```sh
./test_strepsiptera_real_quartets.sh
```-->
