
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