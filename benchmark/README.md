# CNN Benchmark for Pose Classification

This benchmark suite compares various CNN architectures for pose classification.

## Models Included
- VGG16
- VGG19
- ResNet50
- DenseNet121
- DenseNet201
- MobileNetV2
- InceptionV3
- EfficientNetB0

## Directory Structure

benchmark/
├── data/ # Dataset handling
├── models/ # Model architectures
├── utils/ # Utility functions
├── configs/ # Configuration files
├── results/ # Training results
├── train_benchmark.py # Training script
└── evaluate_benchmark.py # Evaluation script


## Usage

1. Training:
```bash
python train_benchmark.py
```

2. Evaluation:
```bash
python evaluate_benchmark.py
```

## Results
Results are saved in the `results` directory with timestamps.
Each run creates:
- Model checkpoints
- Training curves
- Performance comparisons
- Evaluation metrics