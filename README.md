# DegradeMaster
Source Code of ISMB/ECCB'25 submitted paper "Accurate PROTAC targeted degradation prediction with DegradeMaster".

![alt text](https://github.com/Jackson117/DegradeMaster/blob/main/framework.png?raw=true)

## Dependencies
+ python==3.10.13
+ pyg=2.5.2
+ pytorch-cuda=11.6
+ torch==2.4.0 
+ torch-cluster==1.6.3+pt23cu118 
+ torch-scatter==2.1.2+pt23cu118 
+ torch-sparse==0.6.18+pt23cu118 
+ torch-spline-conv==1.2.2+pt23cu118 
+ torchaudio==2.3.0+cu118
+ networkx==3.3
+ numpy==1.23.5
+ scipy==1.10.1

The full dependencies can be installed by executing the command below:
```
conda env create --name envname --file=protac.yml
```

## Usage
### Accurate PROTAC-targeted degradation prediction on datasets crafted from PROTAC-DB 3.0
To train and evaluate on PROTAC-8K:
1. Download the dataset from https://zenodo.org/records/14728925, and paste folders at ./data/PROTAC
2. Execute the command below:
```
python main.py --config config/config.yml
```

### Case study
To conduct the case study #1 for VZ185 candidate degradation prediction:
```
python case_study.py
```

To conduct the case study #2 for ACBI3 on KRAS mutant degradation prediction:
1. Remove all the files in ./data/case_study/processed
```
cd ./data/case_study/processed
rm *.pt
```
2. Change the value of "dataset_type" in ./config/config_c.yml to "case_study_2"

3. Execute the command below:
```
python case_study.py
```

### Use degrademaster through command
Install degrademaster as a package
```
pip install -e .
```

Command options
```
degrademaster --help
PROTAC Degradation Prediction

options:
  -h, --help            show this help message and exit
  --mode {Train,Test}   Run mode
  --dataset DATASET     Dataset JSON name (without extension)
  --data_format DATA_FORMAT
                        The format of input data list
  --seed SEED           Random seed
  --train_rate TRAIN_RATE
                        Train/val split ratio
  --show_input SHOW_INPUT
                        Show input
  --conv_name {GCN,GAT,SAGE,EGNN}
                        Type of graph convolution layer
  --hidden_size HIDDEN_SIZE
                        Hidden dimension size
  --n_layers N_LAYERS   Number of layers (used in EGNN)
  --attention           Use attention in EGNN
  --feature             Use real features instead of random
  --select_pocket_war SELECT_POCKET_WAR
                        Pocket warhead selection threshold
  --select_pocket_e3 SELECT_POCKET_E3
                        Pocket E3 selection threshold
  --e3_dim E3_DIM       Ligase pocket graph feature dimension
  --protac_dim PROTAC_DIM
                        PROTAC graph feature dimension
  --tar_dim TAR_DIM     Target pocket graph feature dimension
  --batch_size BATCH_SIZE
                        Batch size
  --epoch EPOCH         Number of training epochs

```
Make predictions with pretrained model
```
degrademaster --mode Test --dataset TRIM24
```

### Reference
```
@article{liu2025accurate,
  title={Accurate PROTAC targeted degradation prediction with DegradeMaster},
  author={Liu, Jie and Roy, Michael and Isbel, Luke and Li, Fuyi},
  journal={bioRxiv},
  pages={2025--02},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```
