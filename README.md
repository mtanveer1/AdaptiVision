**AdaptiVision: A Flexible and Efficient Vision Transformer for Adaptive Token Pruning**

AdaptiVision is a hierarchical Vision Transformer that adaptively reshapes token representations across stages using two core innovations:

- **CA-DTC** (Coordinate-Aware Dynamic Token Clustering): differentiable soft clustering that jointly uses semantic features and spatial coordinates to produce geometrically grounded super-tokens.
- **AFA** (AdaptiveFocus Attention): pre-attention channel recalibration that amplifies task-relevant channels and suppresses noisy ones before self-attention.

---

## Results

### ImageNet-1K Classification (224×224)

| Model               | Params (M) | GFLOPs | Top-1 (%) |
|---------------------|-----------|--------|-----------|
| AdaptiVision-Light  | 12.8      | 2.6    | 80.2      |
| AdaptiVision-Medium | 25.8      | 4.8    | 82.6      |
| AdaptiVision-Large  | 61.7      | 11.9   | 84.2      |

### ADE20K Semantic Segmentation (Semantic FPN, 512×512)

| Model               | Params (M) | GFLOPs | mIoU (%) |
|---------------------|-----------|--------|----------|
| AdaptiVision-Light  | 26.2      | 14.6   | 46.9     |
| AdaptiVision-Medium | 34.0      | 44.8   | 52.2     |

### COCO-WholeBody Pose Estimation (384×288)

| Model               | GFLOPs | Whole-body AP |
|---------------------|--------|--------------|
| AdaptiVision-Medium | 15.7   | 63.6         |
| AdaptiVision-Large  | 29.6   | 64.1         |

---

## Repository Structure

```
adaptivision/
├── configs/
│   ├── adaptivision_light.yaml
│   ├── adaptivision_medium.yaml
│   └── adaptivision_large.yaml
├── data/
│   ├── __init__.py
│   ├── build.py
│   └── transforms.py
├── models/
│   ├── __init__.py
│   └── adaptivision.py
├── modules/
│   ├── __init__.py
│   ├── ca_dtc.py
│   ├── afa.py
│   └── mix_ffn.py
├── scripts/
│   ├── train_imagenet.sh
│   ├── eval_imagenet.sh
│   └── train_ade20k.sh
├── tools/
│   ├── train.py
│   └── evaluate.py
├── utils/
│   ├── __init__.py
│   ├── checkpoint.py
│   └── logger.py
├── requirements.txt
└── setup.py
```

---

## Installation

```bash
git clone https://github.com/mtanveer1/adaptivision.git
cd adaptivision
pip install -r requirements.txt
pip install -e .
```

---



