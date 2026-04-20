# LIFE703 Research Project
This repository contains the codes and outputs for the MSc Bioinformatics research project on digital camera images cataract detection and classification using convolutional neural network.

# Objectives
- Develop a pre-processing model for image segmentation to enhance CNN model performance
- Select, adapt and train selected CNN models for binary detection and multi-class classification
- Evaluate, compare and recommend suitable models for future development and use

# Background
Cataract is characterised by clouding of the lens and is a leading cause of blindness worldwide. Underdiagnosis remains a major challenge and results in functional impairments, prompting artificial intelligence assisted solutions. Current cataract research heavily focuses on surgical treatment methods while developed detection models utilise medical photographs that are inaccessible for most. Convolution neural networks (CNN) is a feedforward neural network strong in image-based pattern recognition tasks by kernel optimisation. This research adapts existing CNN models to detect and classify cataract cases in digital camera images.

# Dataset
The original dataset for the binary classification is obtained from the train directory of https://github.com/krishnabojha/Cataract_Detection-using-CNN, which contains 4354 cataract and 3714 normal images used for training. The external validation dataset is obtained from https://www.kaggle.com/datasets/nandanp6/cataract-image-dataset, with all 306 cataract and 306 normal images used for testing. Both datasets are pre-processed with a pre-segmentation model trained with 240 manually curated binary image masks by applying a three-phase circle crop to isolate the iris.

# Repository Contents
The `segmentation` folder contains the following subfolders and files:
- `Drawings`: annotated images with the iris colooured in blue
- `Images`: 240 original images selected randomly from both training and testing datasets
- `Masks`: corresponding binary masks of the selected images
- create_masks.py: Python code to convert manually annotated images into masks
- non-segmented.pdf: training results of ResNet50 using the non-segmented original datasets
- pupil_unet_model.h5: segmentation model
- segmentation.py: Python code to perform image segmentation
- segmented.pdf: training results of ResNet50 using segmented datasets

The `optimisation` folder contains the subfolders `densenet`, `mobilenet` and `resnet`. Each subfolder contains the files:
- optimisation.py: Python to code generate graphs for learning rate optimisation
- phase1_lr.png: plot of loss against learning rate for feature extraction (phase 1 with frozen base model)
- phase2_lr.png: plot of loss against learning rate for fine-tuning (phase 2 with unfrozen base model)

The `binary_classification` folder contains the following subfolders and files:
- {model}.pdf: training results of respective models 
- {model}.py: Python code of each model for binary classification using segmented datasets
- `plots`
    - loss_data_{model}.npz: raw data of loss statistics
    - roc_data_{model}.npz: raw data of roc statistics
    - overlay_graphs.py: Python code to generate the overlayed graphs containing statistics from all 3 models
    - ROC_curve.png: comparison of the ROC curve of all 3 models
    - validation_loss.png: comparison of the loss over epochs of all 3  models
 
# Pre-requisites

# Features
