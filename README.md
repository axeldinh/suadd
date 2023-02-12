# AIcrowd Scene Understanding for Autonomous Delivery Challenge

This repository contains the code for the AIcrowd Scene Understanding for Autonomous Delivery Challenge.

## Installation

```
conda env create -f environment.yml
```

## Code Structure

``` 
.
├── README.md
├── environment.yml
├── requirements.txt
├── configs
│   ├── globals.py                    # Contains the global variables (debug, dataset_path), should be modified for each user
│   ├── wandb.py                      # Contains the information for the Weights and Biases logger
│   ├── configs_unet.py               # Contains the configuration using the U-Net model
│   ├── experiments.py                # Contains the configuration loading function
├── models                            # Contains the models
│   ├── __init__.py
│   ├── lightning_module.py           # Contains the general PyTorch Lightning module for training and predictions
│   ├── unet.py
│   └── ...
├── losses                            # Contains the loss functions for training
│   ├── __init__.py
│   ├── cross_entropy_mse.py
│   └── ...
├── transforms                        # Contains the transforms for data augmentation
│   ├── __init__.py
│   ├── transforms_1.py
│   └── ...
├── utils                             # Contains the utilities
│   ├── __init__.py
│   ├── images.py                     # Image processing utils
│   ├── datasets.py                   # Data processing utils
│   ├── metrics.py                    # Metrics utils contains the metrics used for the challenge (such as SiLog)
│   ├── transforms.py                 # Transforms utils, contains modules for data augmentation
│   └── ...
├── datasets                          # Contains the datasets with the images and the annotations, not stored in the repo
│   ├── suadd
│   │   ├── inputs
│   │   │   ├── ...
│   │   ├── semantic_annotations
│   │   │   ├── ...
│   │   └── depth_annotations
│   │       ├── ...
│   └── ...
├── outputs                           # Contains the outputs of the experiments, created when running the code
│   ├── Experiment_<experiment_id>
│   │   ├── trial_<trial_number>
│   │   │   ├── git_wandb_config.txt  # Contains the git commit and the wandb run id 
│   │   │   ├── checkpoints           # Contains the checkpoints of the training
│   │   │   ├── wandb                 # Contains the wandb files
│   │   │   ├── ...
│   │   └── ...      
├── load_data.py                      # Loads the data from wandb and saves it locally
├── check_dataset.py                  # Checks the data and the annotations from an experiment ID and saves the results locally
├── create_dataset_artifact.py        # Creates a dataset artifact from the local data
├── train.py                          # Trains the model using the experiment ID
└── ...
```

## Usage

### Data

#### Load from wandb

To use the data from wandb, you need to have a wandb account and to be logged in. Then, you can run the following
command to download the data:

```
python load_data.py
```

Please note that the full data cannot be stored using W&B, so only a subset of the data is stored.

The dataset should be stored in the path specified in ``configs/globals.py``. The dataset should be stored in the
following structure:

```
.
├── dataset_name
│   ├── inputs
│   │   ├── image1.png
│   │   ├── image2.png
│   │   ├── ...
│   ├── semantic_annotations
│   │   ├── image1.png
│   │   ├── image2.png
│   │   ├── ...
│   └── depth_annotations
│       ├── image1.png
│       ├── image2.png
│       ├── ...
└── ...
```

#### Create dataset artifact

To create a dataset artifact, you need to have a wandb account and to be logged in. Then, you can run the following
command to create the dataset artifact:

```
python create_dataset_artifact.py
```

#### Check dataset

To check the transformations applied to the dataset, you can run the following command:

```
python check_dataset.py --exp_id <experiment_id>
```

where ``<experiment_id>`` is the ID of the experiment you want to check. The results will be saved in
the ``check_dataset`` folder.

### Training

First note that the training is done has follows:

- The data is split more or less equally between the training, validation and test sets so that the classes are well
  represented in each set.
- The data is augmented using the transforms specified in the configuration file.
- For training samples, the data is augmented using the transforms specified in the configuration file.
- For validation and test, as we want to fully evaluate the images, we might have to patch the images to make them
  fit the model input size. Hence, the transform module applies specific transforms for validation and test (both before
  inference and for reconstruction in case of patches).
- The model is trained using the PyTorch Lightning framework. The training is done using the ``train.py`` script.
- The training is done using the configuration file specified in the ``--exp_id`` argument. The configuration file
  contains the model, the loss function, the optimizer, the data augmentation transforms, the training parameters, ...
- The metrics are computed both globally and per class.

To train a model, you can run the following command:

```
python train.py --exp_id <experiment_id> --trial_id <trial_number>
```

where ``<experiment_id>`` is the ID of the experiment you want to train. 
``<trial_number>`` is the trial number (if you want to train the model multiple times or resume a previous run).
The results will be saved in
the ``outputs/Experiment_<experiment_id>/trial_<trial_number>`` folder. If the ``--trial`` argument is not specified,
the training will automatically define a new trial number.

#### Resume training

To resume a training from a checkpoint, you can run the following command:

```
python train.py --exp_id <experiment_id> --trial_id <trial_number>
```

if the run for the specified ``<experiment_id>`` and ``<trial_number>`` has already been started, the training will
automatically resume from the last checkpoint. To do so, the wandb run id is saved in the ``git_wandb_config.txt`` file,
so you need to make sure that the file is present in the ``outputs/Experiment_<experiment_id>/trial_<trial_number>``.
In case of issues, check that WandB contains the run with the specified ID.


# Steps

- [x] Prepare the environment
- [ ] Test a U-Net model
- [ ] Reproduce ``Fully Convolutional Networks for Semantic Segmentation`` ([ArXiv](https://arxiv.org/abs/1411.4038)).
- [ ] Test Transformer-Based Models

# Ideas

## Data Augmentation:

- Randomly rotate the image
- Apply random perspective transform
- Brightness and contrast augmentation
- Shifts
- Flips
- Rescaling
- Random elastic deformation
- Random noise
- Random blur

## Models:

### Semantic Segmentation:

- [ ] Test models from torchvision, huggingface, ... Take the best one
- [ ] Use both semantic segmentation and depth estimation as one training task -> Need a single loss function
- [ ] U-Net, FCN, Mask R-CNN, DeepLabV3

### Depth Estimation:

- [ ] If tag accuracy good enough, can use it at inference time to improve depth estimation (get the size of the square
  and deduce the distance)

# TODOs

- [x] Split Data (make sure it is balanced)
- [x] Data augmentation
- [x] Prepare PyTorch Lightning model (use debug mode)
- [x] Prepare WandB logging (use debug mode)
- [x] Do not forget to keep track of commit hashes for reproducibility
- [x] Prepare a training script for Google Colab
- [ ] Start training
- [ ] allow th framework to work only on depth estimation