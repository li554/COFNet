## <div align="center">COFNet: Contrastive Object-aware Fusion using Box-level Masks for Multispectral Object Detection</div>

### Introduction

In this paper, we propose an innovative multispectral object detection method that combines contrastive learning and a new cross-modal feature fusion module. We introduce a mask feature contrastive loss that maximizes the similarity between the box-level mask features and modal features while suppressing background responses, enabling effective representative alignment between the input and output spaces. Additionally, we propose a mask-guided attention fusion module that uses predicted mask features to guide the fusion of different modal features, enhancing object responses and reducing background noise interference.


### Installation

Clone repo and install requirements.txt in a Python>=3.8.0 conda environment, including PyTorch>=1.12.

```
git clone https://github.com/li554/COFNet.git
cd COFNet
pip install -r requirements.txt
```

### Weights

- **FLIR-aligned**
  Link：https://pan.baidu.com/s/1NYOhvxqGBZfD-Vg3zLQx5g?pwd=fr75 
  Code：fr75
### Testing

```
python test_wb.py
```
### Citation
  waiting for update...
