# NAS-PED: Neural Architecture Search for Pedestrian Detection

This repository contains the official implementation of **NAS-PED: Neural Architecture Search for Pedestrian Detection**.

```

## Installation

### 1. Create Conda Environment

```bash
conda create -n pednas python=3.8
conda activate pednas
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
```

### 2. Install MMCV

#### For DDQ FCN & R-CNN (NAS-PED-NAS_PED)

```bash
cd NAS-PED-NAS_PED/mmcv-1.4.7
MMCV_WITH_OPS=1 python setup.py build_ext --inplace
ln -s mmcv-1.4.7/mmcv ./
cd ..
export PYTHONPATH=`pwd`:$PYTHONPATH
```

#### For DDQ DETR (NAS-PED-NAS_PED_DETR)

```bash
cd NAS-PED-NAS_PED_DETR/mmcv-2.0.0rc4
MMCV_WITH_OPS=1 python setup.py build_ext --inplace
ln -s mmcv-2.0.0rc4/mmcv ./
cd ..
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

## Dataset Preparation

### CrowdHuman

```
data/
├── crowdhuman/
│   ├── annotations/
│   │   ├── annotation_train.odgt
│   │   └── annotation_val.odgt
│   ├── Images/
│   │   ├── train/
│   │   └── val/
```

### EuroCityPersons (ECP)

```
data/
├── ECP/
│   ├── annotations/
│   ├── day/
│   │   ├── img/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── labels/
│   │       ├── train/
│   │       └── val/
```

## Training

### DDQ FCN with NAS-PED-S on CrowdHuman

```bash
cd NAS-PED-NAS_PED
sh tools/train.sh projects/configs/ddq_fcn/ddq_fcn_nas_ped_s_3x_crowdhuman.py 8 ./work_dirs/ddq_fcn_nasped_s_ch
```

### DDQ FCN with NAS-PED-S on ECP

```bash
cd NAS-PED-NAS_PED
sh tools/train_ecp.sh projects/configs/ddq_fcn/ddq_fcn_nas_ped_s_3x_ecp.py 8 ./work_dirs/ddq_fcn_nasped_s_ecp
```

### DDQ R-CNN with NAS-PED on CrowdHuman

```bash
cd NAS-PED-NAS_PED

# NAS-PED-T
sh tools/train.sh projects/configs/ddq_rcnn/ddq_rcnn_nas_ped_t_3x_crowdhuman.py 8 ./work_dirs/ddq_rcnn_nasped_t_ch

# NAS-PED-S
sh tools/train.sh projects/configs/ddq_rcnn/ddq_rcnn_nas_ped_s_3x_crowdhuman.py 8 ./work_dirs/ddq_rcnn_nasped_s_ch

# NAS-PED-B
sh tools/train.sh projects/configs/ddq_rcnn/ddq_rcnn_nas_ped_b_3x_crowdhuman.py 8 ./work_dirs/ddq_rcnn_nasped_b_ch
```


### For Slurm Cluster

```bash
GPUS=8 sh tools/slurm_train.sh partition_name job_name projects/configs/ddq_rcnn/ddq_rcnn_nas_ped_s_3x_crowdhuman.py ./work_dirs/ddq_rcnn_nasped_s_ch
```

## Testing

### DDQ FCN with NAS-PED-S

```bash
cd NAS-PED-NAS_PED

# Test on CrowdHuman
sh tools/test.sh projects/configs/ddq_fcn/ddq_fcn_nas_ped_s_3x_crowdhuman.py path_to_checkpoint 8 --eval bbox

# Test on ECP
sh tools/test_ecp.sh projects/configs/ddq_fcn/ddq_fcn_nas_ped_s_3x_ecp.py path_to_checkpoint 8 --eval bbox
```

### DDQ R-CNN with NAS-PED

```bash
cd NAS-PED-NAS_PED

# NAS-PED-T
sh tools/test.sh projects/configs/ddq_rcnn/ddq_rcnn_nas_ped_t_3x_crowdhuman.py ../DDQ_RCNN_NASPED_T_CH_4e65de52.pth 8 --eval bbox

# NAS-PED-S
sh tools/test.sh projects/configs/ddq_rcnn/ddq_rcnn_nas_ped_s_3x_crowdhuman.py ../DDQ_RCNN_NASPED_S_CH_4ae54a56.pth 8 --eval bbox

# NAS-PED-B
sh tools/test.sh projects/configs/ddq_rcnn/ddq_rcnn_nas_ped_b_3x_crowdhuman.py ../DDQ_RCNN_NASPED_B_CH_ff52b7e1.pth 8 --eval bbox
```


### For Slurm Cluster

```bash
GPUS=8 sh tools/slurm_test.sh partition_name job_name projects/configs/ddq_rcnn/ddq_rcnn_nas_ped_s_3x_crowdhuman.py path_to_checkpoint --eval bbox
```

## Results and Models

### ImageNet Pretrained Backbones

| Backbone | Download |
|:--------:|:--------:|
| NAS-PED-T | [model](https://drive.google.com/file/d/1gM8iH0tf83TCCSZlw0QQkZP5EolWmurs/view?usp=sharing) |
| NAS-PED-S | [model](https://drive.google.com/file/d/1N7uNnk3SXfJtoUKI1RzVK4f4SiYLx-fO/view?usp=sharing) |
| NAS-PED-B | [model](https://drive.google.com/file/d/1U6s53w0VBg3rUg0ByC9vjDn_t4XKoMQx/view?usp=sharing) |

### DDQ R-CNN on CrowdHuman

| Backbone | Config | Download |
|:--------:|:------:|:--------:|
| NAS-PED-T | [config](projects/configs/ddq_rcnn/ddq_rcnn_nas_ped_t_3x_crowdhuman.py) | [model](https://drive.google.com/file/d/1dGbfEVK_DxMh0D__8GnVElfcOFREmE9k/view?usp=sharing) |
| NAS-PED-S | [config](projects/configs/ddq_rcnn/ddq_rcnn_nas_ped_s_3x_crowdhuman.py) | [model](https://drive.google.com/file/d/1v-TU3AB5aSjW-OCwY2X7wNBa2TRV5s4E/view?usp=sharing) |
| NAS-PED-B | [config](projects/configs/ddq_rcnn/ddq_rcnn_nas_ped_b_3x_crowdhuman.py) | [model](https://drive.google.com/file/d/1cvzTNYL_Q35QCP_dTG5BpDAuCQjdDnJr/view?usp=sharing) |

### DDQ FCN on EuroCityPersons (ECP)

| Backbone | Config | Download |
|:--------:|:------:|:--------:|
| NAS-PED-S | [config](projects/configs/ddq_fcn/ddq_fcn_nas_ped_s_3x_ecp.py) | [model](https://drive.google.com/file/d/1kFvMYaBjKhVCO64auuyFFiviSXiojDKs/view?usp=sharing) |

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{tang2025NAS,
  title={NAS-PED: Neural Architecture Search for Pedestrian Detection},
  author={Tang, Yi and Liu, Min and Li, Baopu and Wang, Yaonan and Ouyang, Wanli},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={47},
  number={3},
  pages={1800--1817},
  year={2025},
  publisher={IEEE}
}
```

## Acknowledgements

This codebase is built upon [DDQ](https://github.com/jshilong/DDQ), [MMDetection](https://github.com/open-mmlab/mmdetection), and [MMCV](https://github.com/open-mmlab/mmcv). We thank the authors for their excellent work.

## License

This project is released under the Apache 2.0 license.
