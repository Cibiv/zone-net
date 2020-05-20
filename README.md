# Distinguishing Felsenstein zone from Farris zone

## Prerequisites

For all scripts Python 3.6.6 was used. The python packages used can be installed via
```sh
pip install (--user) -r packages_required.txt
```
if python3 and pip are already installed.

To pull the raw data from this repository, please make sure that you have [Git Large File Storage](https://git-lfs.github.com/) installed.

## Network for simulated alignments using Jukes-Cantor model

The results presented in the manuscript stem from the network `F-zoneNN_2` saved in the models folder. This model can be retrained by first preprocessing the training data via 
```sh
./preprocess_zone_train_data.sh
```
in the folder data/preprocessing and then training the network using `config_F-zoneNN.yaml` in the config folder
```sh
python3 mlp.py config/config_F-zoneNN.yaml
```
If an already trained network should be tested, first the test data has to be preprocessed by
```sh
./preprocess_zone_test_data.sh
```
in the folder data/preprocessing.

Testing of the network can be done by executing:
```sh
python3 test_zone.py [-h]
```

## Network for Strepsiptera data

The results presented in the manuscript stem from the network `StrepsipteraNN_3` saved in the models folder. The preprocessing for the training data
```sh
./preprocess_strepsiptera_train_data.sh
```
and for the test data
```sh
./preprocess_strepsiptera_test_data.sh
```
can both be started in the folder data/preprocessing. The same holds for the extraction of the frequencies from the Strepsiptera quartets
```sh
./preprocess_strepsiptera_real_quartets.sh
```
as well as for the extraction of the frequencies from the differently ordered Strepsiptera quartets
```sh
./preprocess_strepsiptera_real_quartets_permuted.sh
```
The network can then be trained via
```sh
python3 mlp.py config/config_StrepsipteraNN.yaml
```
To extract for which strepsiptera quartets the network infers a Farris-type or Felsenstein-type tree excecute
```sh
python3 test_strepsiptera_quartets.py [-h]
```
The output lists the number of permutations of a quartet for which the network infers a Farris-type or Felsenstein-type tree.
Testing of the network on simulated data can be done by executing:
```sh
python3 test_simulated_strepsiptera_alignments.py [-h]
```

