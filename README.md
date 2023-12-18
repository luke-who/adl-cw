# Acoustic Scene Classification with Convolutional Neural Network (CNN)
-----------------------------------------------------------------------------------
[![python](https://img.shields.io/badge/python-3.7.3-blue?style=plastic&logo=python)](https://www.python.org/downloads/release/python-373/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10-orange?logo=PyTorch)](https://github.com/pytorch/pytorch/releases/tag/v1.10.0)

Before running the code on [BC4](https://www.bristol.ac.uk/acrc/high-performance-computing/hpc-systems-tech-specs/), make sure the "ADL_DCASE_DATA" is unzipped into the [`data`](data) folder.
# Data path:
```
data
└── ADL_DCASE_DATA
    ├── development
    │   ├── audio/(1170 '*.npy' files)
    │   └── labels.csv
    └── evaluation
        ├── audio/(390 '*.npy' files)
        └── labels.csv
 ```       
# How to run the code on [BC4](https://www.bristol.ac.uk/acrc/high-performance-computing/hpc-systems-tech-specs/):
`$ cd work`

`$ sbatch train_dcase.sh`





**Note the data was not altered in this experiment, it's the same as the original [DCASE](http://dcase.community/challenge2017/download)**
