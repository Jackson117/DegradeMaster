# DegradeMaster
Source Code of ISMB/ECCB'25 submitted paper "Accurate PROTAC targeted degradation prediction with DegradeMaster"

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

### Reference
```bibtex
@inproceedings{liu2024bourne,
  title={Bourne: Bootstrapped self-supervised learning framework for unified graph anomaly detection},
  author={Liu, Jie and He, Mengting and Shang, Xuequn and Shi, Jieming and Cui, Bin and Yin, Hongzhi},
  booktitle={2024 IEEE 40th International Conference on Data Engineering (ICDE)},
  pages={2820--2833},
  year={2024},
  organization={IEEE}
}
```
