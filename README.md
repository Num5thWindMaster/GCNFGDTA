# Resources
## KIBA dataset:
Please jump to https://doi.org/10.1021/ci400709d in order to get KIBA dataset.
## Compounds of _Cyperus esculentus_
We provide CAS Number of each compound. Please refer to molecule_data/CAS_name.txt

## Source codes:
initial/get_dict.py: Match function groups from pre-trained model and transfer to embedding weights of function groups.
models/ : including stem models and the function group enhance model.
targets_screen/ : including molecule screening tool scripts.
predict_data/ : including tool scripts for predicting target compounds.
creat_data.py: create data in pytorch format
utils.py: include TestbedDataset used by create_data.py to create data, and performance measures.
training_validation.py: train a GCNFG-DTA model.

# Running the model de novo: 
## 0. Move KIBA dataset to data/, including data/kiba_test.csv and data/kiba_train.csv

## 1. Install Python libraries needed:

```
conda create -n GCNFGDTA python=3.9
conda activate GCNFGDTA
pip install -r requirements.txt
```

## 2. Create Data:
```
python create_data.py
python ./initial/get_dict.py
```

## 3. Train a prediction model
```
python training_validation.py 1 6 0
```

## 4. Predicting affinity scores of molecule-target pairs 
Please edit pairs in predict_data/test.txt. Each row with a pair, using _space_ to split SMILES and amino acid sequence. 
```
python training_validation.py 1 6 0
```
please attention parameters '1 6 0' should be same with step 3.