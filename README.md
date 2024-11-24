# Resources
## KIBA dataset:
Please jump to https://doi.org/10.1021/ci400709d in order to get KIBA dataset.
## BindingDB dataset:
Please jump to https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?download_file=/bind/downloads/BindingDB_ChEMBL_202411_tsv.zip to download BindingDB dataset.
## Compounds of _Cyperus esculentus_
We provide CAS Number of each compound. Please refer to molecule_data/CAS_name.txt

## Source codes:
initial/get_dict.py: Match function groups from pre-trained model and transfer to embedding weights of function groups.

models/ : Including stem models and the function group enhance model.

targets_screen/ : Including molecule screening tool scripts.

metrics/test_auroc.py: plot auroc curves for selected models and datasets.

predict_data/ : Including tool scripts for predicting target compounds.

fgp/ : Function group enhance data, run each dataset will create a .pt file by once.

creat_data.py: Create data in pytorch format

utils.py: Including TestbedDataset used by create_data.py to create data, and utils affiliate to main model.

training_validation.py: train the GCNFG-DTA model, or train the Graph-DTA baseline.

training_validation_cv.py: train the GCNFG-DTA model using n-folds cross validation. 
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
python training_validation.py 1 6 0 0
```

## 4. Acquiring model metrics 
Please edit pairs in predict_data/test.txt. Each row with a pair, using _space_ to split SMILES and amino acid sequence. 
```
python training_validation.py 1 6 0 -1
```
please note parameters '1 6 0' should be same with step 3.

## 5. Predicting affinity scores of molecule-target pairs 
Please edit pairs in predict_data/test.txt. Each row with a pair, using _space_ to split SMILES and amino acid sequence. 
```
python predicting.py 1 6 0
```
please note parameters '1 6 0' should be same with step 3.