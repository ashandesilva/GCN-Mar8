# Human Pose Classification using Graph Convolutional Networks

This project implements a Graph Convolutional Network (GCN) for classifying human poses in barbell press exercises, specifically detecting lumbar errors.

## Project Structure
```
.
├── data_v1/                  # Contains image frames
│   ├── lumbar_3_7_dell_111/ # Frames with lumbar errors
│   └── Correct_n2_3_7_100/  # Frames with correct form
├── annotations_v1/           # Contains keypoint annotations
├── src/
│   ├── data_processing.py   # Data loading and preprocessing
│   ├── gcn_model.py         # GCN model architecture
│   ├── train.py             # Training script
│   └── utils.py             # Utility functions
└── requirements.txt         # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Format

- 17 keypoint annotations per frame
- Two classes: correct form and lumbar error
- Image naming convention: videoID_frameNumber (e.g., 52701_11_9)

## Training

To train the model:
```bash
python src/train.py
```

## Model Architecture

The GCN model consists of:
1. Graph construction from 17 keypoints
2. Multiple GCN layers for feature extraction
3. Global pooling and fully connected layers for classification

## Dependencies

- PyTorch
- PyTorch Geometric (torch-geometric)
- NumPy
- OpenCV
- JSON 




### Summary the complete GCN-based solution for your human pose classification task :

1. **Project Structure**:
   - `src/data_processing.py`: Handles data loading and preprocessing
   - `src/gcn_model.py`: Contains the GCN model architecture
   - `src/train.py`: Training and evaluation script
   - `src/utils.py`: Helper functions for visualization and inference
   - `requirements.txt`: Required dependencies
   - `README.md`: Project documentation

2. **Key Features**:
   - Custom GCN architecture with 3 graph convolutional layers
   - Dropout for regularization
   - Binary classification (correct vs. incorrect form)
   - Training with validation metrics (accuracy, precision, recall, F1)
   - Visualization tools for poses and training metrics
   - Model checkpointing (saves best model)

3. **How to Use**:
   1. Set up the environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

   2. Train the model:
   ```bash
   python src/train.py
   ```

   The model will train for 100 epochs and save:
   - Best model weights as `best_model.pth`
   - Training metrics plot as `training_metrics.png`

4. **Technical Details**:
   - The GCN processes 17 keypoints as nodes
   - Each node has 2 features (x, y coordinates)
   - The skeleton connections are defined as edges in the graph
   - Data is automatically split into 80% training and 20% validation
   - Uses Adam optimizer with learning rate 0.001
   - Includes data normalization and augmentation

5. **Visualization**:
   - Training progress is displayed with metrics
   - Pose visualization function for debugging
   - Training curves are plotted and saved

#### To improve the model's performance, Future work:
1. Adjust the model architecture (add/remove layers, change layer sizes)
2. Modify training parameters (learning rate, batch size, number of epochs)
3. Add data augmentation techniques
4. Experiment with different graph edge connections
5. Try different optimizers or learning rate schedules
