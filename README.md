# CPM-TransDetector Training Codes

## English

This repository provides the training codes for the CPM-TransDetector experiments.

The Python codes are used to train the Transformer-based CPM detector for binary and quaternary CPM signals. The two-ray and six-ray channel cases use the same training pipeline, and the corresponding experiment can be selected by changing the input dataset path.

### File Description

| File | Description |
|---|---|
| `train_binary_cpm.py` | Training code for binary CPM detection. The two-ray and six-ray channel cases can be selected by changing the dataset path. |
| `train_quaternary_cpm.py` | Training code for quaternary CPM detection. The two-ray and six-ray channel cases can be selected by changing the dataset path. |
| `cpm_train_utils.py` | Common training utilities, including metric accumulation and auxiliary functions. |

### Notes

- The Python scripts are used to train the Transformer-based CPM detector.
- Separate training files are provided for binary and quaternary CPM.
- The two-ray and six-ray channel cases use the same training procedure.
- Different channel cases can be selected by changing the corresponding dataset path.

---

## 中文说明

本仓库提供 CPM-TransDetector 相关实验的训练代码。

Python 代码用于训练面向二进制和四进制 CPM 信号的 Transformer 检测器。二径和六径信道实验采用相同的训练流程，可通过修改输入数据集路径选择相应的实验数据。

### 文件说明

| 文件 | 说明 |
|---|---|
| `train_binary_cpm.py` | 二进制 CPM 检测训练代码。可通过修改数据集路径选择二径或六径信道实验。 |
| `train_quaternary_cpm.py` | 四进制 CPM 检测训练代码。可通过修改数据集路径选择二径或六径信道实验。 |
| `cpm_train_utils.py` | 通用训练工具函数，包括指标统计和辅助函数。 |

### 说明

- Python 脚本用于训练基于 Transformer 的 CPM 检测器。
- 二进制 CPM 和四进制 CPM 分别提供独立的训练代码。
- 二径和六径信道实验采用相同的训练流程。
- 可通过修改对应的数据集路径选择不同的信道实验。
