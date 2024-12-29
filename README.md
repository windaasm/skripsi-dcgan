# DCGAN and GoogLeNet for Rice Leaf Disease Detection

## Overview
This repository contains implementations of two deep learning approaches to tackle the problem of rice leaf disease detection:

- **DCGAN (Deep Convolutional Generative Adversarial Network):** Used for generating synthetic rice leaf images to augment the dataset.
- **GoogLeNet:** Used for classifying rice leaf diseases based on the augmented dataset.

The project leverages both TensorFlow and PyTorch to build and train the models.

This work was part of my undergraduate thesis project.

## Dataset
The dataset used in this project is sourced from Kaggle:
[**Rice Leaf Diseases Dataset**](https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases/code)

It contains images of rice leaves affected by various diseases, categorized into different classes.

## Models

### 1. DCGAN
- **Purpose:** Augment the dataset by generating synthetic images of rice leaves.
- **Libraries Used:** TensorFlow
- **Features:**
  - Generator and Discriminator models implemented from library TensorFlow.
  - Training pipeline with adjustable parameters such as learning rate and batch size.

### 2. GoogLeNet
- **Purpose:** Classify rice leaf images into their respective disease categories.
- **Libraries Used:** Pytorch
- **Features:**
  - Fine-tuned GoogLeNet model for high accuracy.
  - Transfer learning for faster convergence.

## Usage

### Prerequisites
- Python 3.x
- TensorFlow
- PyTorch
- NumPy
- Matplotlib
- OpenCV


3. **Prepare the Dataset:**
   - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases/code).
   - Place the images in the `data/` directory.

## Thesis Journal
The complete details of this project, including methodologies, experiments, and results, are documented in the following journal:

- **Title:** Detection of Rice Leaf Diseases using DCGAN and GoogLeNet
- **Link:** [soon]

## Contributions
Feel free to fork and contribute to this project. Suggestions for improvement are always welcome!

