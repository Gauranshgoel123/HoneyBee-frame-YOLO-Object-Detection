
# Bee Frame Segmentation

This repository contains the implementation of a segmentation model aimed at identifying and extracting the actual bee frame from images, removing unwanted elements. The project utilizes state-of-the-art deep learning techniques for semantic segmentation and has been developed with collaboration best practices using GitHub.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Team Collaboration](#team-collaboration)
- [Getting Started](#getting-started)
- [License](#license)

## Overview

The goal of this project is to build a segmentation model that can accurately define the region of the bee frame while removing other elements. We experimented with architectures like U-Net, DeepLabV3+, and Mask R-CNN, using metrics such as mAP, F1-score, and IoU for performance evaluation. The repository is structured to support collaboration, reproducibility, and further optimization.

## Dataset

The dataset used in this project is available at the following location:  

https://docs.google.com/document/d/1EPTBlsVZWjzm_UtPhoRm4iwpkOlmyXhaHJsibw2q25w/edit?usp=sharing
[DeepBee Dataset](https://github.com/avsthiago/deepbee-source/tree/release-0.1/src/data/resources)

Please ensure you download and place the dataset in the `data/raw/` directory before running the preprocessing scripts.

## Data Preprocessing

Data preprocessing includes:
- **Data Augmentation:** Rotation, flipping, brightness adjustments, etc.
- **Normalization:** Scaling pixel values.
- **Resizing:** Uniform image dimensions for model compatibility.
- **Annotation Preparation:** Creating masks for segmentation if not provided.

Detailed scripts are available in the `src/preprocessing/` folder.

## Model Architecture

We experimented with the following segmentation models:
- **U-Net**
- **DeepLabV3+**
- **Mask R-CNN**

The chosen model is implemented in `src/models/` with a configurable architecture that can be fine-tuned. The model uses a combination of Dice Loss and IoU Loss for improved segmentation accuracy.

## Training

The training pipeline is defined in `src/models/train.py`. Key features include:
- **Transfer Learning:** Leveraging pre-trained weights.
- **Hyperparameter Tuning:** Configurable parameters such as learning rate, batch size, and epochs.
- **Checkpointing:** Saving model weights for the best performing epoch.

To run the training, execute:
```bash
python src/models/train.py --config configs/train_config.yaml
```

## Evaluation

Model performance is evaluated using:
- **mAP (Mean Average Precision)**
- **F1-score**
- **IoU (Intersection over Union)**

Evaluation scripts and metrics calculations are located in `src/evaluation/`. Visualizations such as segmentation masks and precision-recall curves are generated to assess the performance.

## Results

A summary of the model’s performance and optimization details can be found in the [reports](reports/) directory. Key results include:
- Improved segmentation accuracy with data augmentation.
- Optimization insights for better inference speed and reduced model size.

## Team Collaboration

- **Version Control:** Managed via GitHub with regular commits and pull requests.
- **Task Division:** Clear separation of data preprocessing, model development, training, and evaluation tasks.
- **Documentation:** Code and process are well-documented throughout the repository.

## Getting Started

### Prerequisites
- Python 3.7+
- PyTorch / TensorFlow (depending on the chosen model framework)
- Other dependencies as listed in `requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bee-frame-segmentation.git
   cd bee-frame-segmentation
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Download the dataset and place it in the `data/raw/` directory.

4. Run data preprocessing:
   ```bash
   python src/preprocessing/preprocess.py --input data/raw --output data/processed
   ```

5. Train the model:
   ```bash
   python src/models/train.py --config configs/train_config.yaml
   ```

6. Evaluate the model:
   ```bash
   python src/evaluation/evaluate.py --config configs/eval_config.yaml
   ```

## License

This project is licensed under the [MIT License](LICENSE).
