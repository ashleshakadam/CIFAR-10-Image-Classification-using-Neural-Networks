# CIFAR-10 Image Classification Using Neural Networks

## Overview
This repository contains an end-to-end implementation of a convolutional neural network (CNN) trained on the CIFAR-10 dataset for multi-class image classification. It demonstrates fundamental deep learning workflows, including data preprocessing, model construction, training with augmentation, evaluation, and deployment considerations.

## Key Features
- **Data Loading & Visualization**: Load CIFAR-10 data; visualize sample images.
- **Preprocessing Pipeline**: Normalize pixel values, one-hot encode labels, and construct an efficient `tf.data` pipeline.
- **Data Augmentation**: Real-time image transformations (rotations, shifts, flips) to improve generalization.
- **CNN Architecture**: Stacked `Conv2D` + `BatchNormalization` + `MaxPooling` blocks, followed by fully connected layers with `Dropout`.
- **Training Strategy**: Adaptive optimizer (Adam), early stopping, learning-rate scheduling, and model checkpointing.
- **Evaluation & Metrics**: Plot training/validation curves, compute test accuracy/loss, and generate confusion matrices.
- **Interpretability**: Grad-CAM saliency maps to visualize decision regions.
- **Deployment**: Guidelines for model exporting, quantization, and inference on edge devices.

## Repository Structure
```
├── data/                   # CIFAR-10 raw and preprocessed data scripts
├── notebooks/              # Jupyter notebooks for exploratory analysis and tutorials
│   └── cifar10_cnn.ipynb   # Step-by-step implementation with visualizations
├── src/                    # Source code modules
│   ├── models.py           # Model architectures and utility functions
│   ├── train.py            # Training loop with callbacks and logging
│   ├── evaluate.py         # Evaluation scripts and metrics generation
│   └── utils.py            # Data loading, augmentation, and preprocessing
├── results/                # Trained model weights, figures, and reports
├── requirements.txt        # Python package dependencies
└── README.md               # Project overview (this file)
```

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/cifar10-cnn.git
   cd cifar10-cnn
   ```
2. Create and activate a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Exploratory Analysis**: Open `notebooks/cifar10_cnn.ipynb` in JupyterLab or Jupyter Notebook to follow the guided walkthrough.
2. **Training**: From the project root, run:
   ```bash
   python src/train.py --epochs 50 --batch-size 64 --learning-rate 1e-3
   ```
   - Checkpoints and logs will be saved to `results/` by default.
3. **Evaluation**: Generate test metrics and visualizations:
   ```bash
   python src/evaluate.py --model-path results/best_model.h5 --output-dir results/
   ```
4. **Interpretability**: Produce Grad-CAM heatmaps:
   ```bash
   python src/utils.py gradcam --image-path data/sample.png --model-path results/best_model.h5
   ```

## Results
- **Test Accuracy**: ~81.5%
- **Test Loss**: ~0.55
- **Best Practices**: Model training stabilized using early stopping (patience=10) and data augmentation.

## Potential Extensions
- Experiment with deeper architectures (ResNet, DenseNet).
- Integrate advanced augmentation policies (CutMix, AutoAugment).
- Deploy model using TensorFlow Lite or ONNX for edge inference.

## Acknowledgments
- CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- TensorFlow and Keras communities for comprehensive documentation and examples.

---

*Prepared by Ashlesha Kadam, The University of Texas at Dallas.*
