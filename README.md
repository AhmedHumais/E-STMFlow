# ESTMFlow

This repository contains the code for the CVPR Workshop paper: ["Spatio-Temporal State Space Model For Efficient Event-Based Optical Flow"](https://openaccess.thecvf.com/content/CVPR2025W/EventVision/papers/Humais_Spatio-Temporal_State_Space_Model_For_Efficient_Event-Based_Optical_Flow_CVPRW_2025_paper.pdf), designed for flow estimation from event-based data. Below are the instructions to set up the environment, prepare the dataset, and run the model for evaluation and training.

## Dependencies and Environment Setup

The code is tested with CUDA 12.1. To set up the environment, follow the steps below:

1. Create a new conda environment with Python 3.12:

    ```bash
    conda create -n estmflow python=3.12
    ```

2. Activate the environment:

    ```bash
    conda activate estmflow
    ```

3. Install dependencies via conda and pip:

    ```bash
    conda install lightning -c conda-forge
    pip install -r requirements.txt
    ```

## Data Preparation

### Training Data

Download the DSEC training dataset and organize it in the following directory structure. The data is publicly available online.
```
DSEC_Dataset 
└── train 
    ├── thun_00_a 
    │ ├── events 
    │ │ ├── left 
    │ │ │ ├── events.h5 
    │ │ │ └── rectify_map.h5 
    │ ├── flow 
    │ │ ├── forward_timestamps.txt 
    │ │ ├── forward 
    │ │ │ ├── 000134.png
```

### Test Data

For the DSEC test data, use the following directory structure. The data is also publicly available online.

```
DSEC_Dataset 
└── test 
    ├── interlaken_00_b 
    │ ├── events_left 
    │ │ ├── events.h5 
    │ │ └── rectify_map.h5 
    │ ├── image_timestamps.txt 
    │ └── test_forward_flow_timestamps.csv
```

## DSEC Evaluation

To generate results using the trained model, run the following command:

```bash
python test.py
```
The results will be saved in the ESTMFlowResults directory. The results can be uploaded to online DSEC flow benchmark to obtain quantitative evaluation metrics. 

## Ablation Training

To reproduce the training results from Table 2 of the paper, run the following command:

```bash
python train_estmflow.py --config [config_filename]
```

Replace [config_filename] with one of the following options:

- m_x2 for Mamba with temporal encoding
- m_x2_p for Mamba with temporal + positional encoding
- t_x2 for Transformer with temporal encoding
- t_x2_p for Transformer with temporal + positional encoding

## Citation

If you find this repository helpful, please cite our paper:

```bibtex
@inproceedings{humais2025stssm,
  title={Spatio-Temporal State Space Model For Efficient Event-Based Optical Flow},
  author={Humais, Muhammad Ahmed and Huang, Xiaoqian and Sajwani, Hussain and Javed, Sajid and Zweiri, Yahya},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={5082--5091},
  year={2025}
}
```

