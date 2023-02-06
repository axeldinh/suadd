
# AIcrowd Scene Understanding for Autonomous Delivery Challenge

This repository contains the code for the AIcrowd Scene Understanding for Autonomous Delivery Challenge.

## Installation

```
conda env create -f environment.yml
```

# Ideas

## Data Augmentation:

- [ ] Randomly rotate the image
- [ ] Apply random perspective transform
- [ ] Brightness and contrast augmentation
- [ ] Shifts
- [ ] Flips
- [ ] Rescaling

## Models:

### Semantic Segmentation:

- [ ] Test models from torchvision, huggingface, ... Take the best one
- [ ] Use both semantic segmentation and depth estimation as one training task -> Need a single loss function

# TODOs

- [ ] Prepare data (balanced classes, train/val/test split, data augmentation)
- [ ] Prepare PyTorch Lightning model (use debug mode)
- [ ] Prepare WandB logging (use debug mode)
- [ ] Do not forget to keep track of commit hashes for reproducibility
- [ ] Prepare a training script for Google Colab
- [ ] Start training