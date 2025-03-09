import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report
import os
import numpy as np
from torch_geometric.data import Data

from src.data_processing import load_dataset, PoseDataset
from src.gcn_model import PoseGCN

def load_trained_model(model_path, device):
    """Load the trained model from a state dict file."""
    # Initialize model
    model = PoseGCN(num_node_features=2).to(device)
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Model loaded successfully")
    
    return model

def prepare_single_pose_input(keypoints, visibility=None):
    """
    Prepare a single pose input for prediction.
    
    Args:
        keypoints: numpy array or list of shape [17, 2] containing x,y coordinates
                  The keypoints should be in COCO format with 17 keypoints in the following order:
                  [nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder,
                   left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip,
                   left_knee, right_knee, left_ankle, right_ankle]
        visibility: optional numpy array or list of shape [17] containing visibility flags
                   0: not labeled, 1: labeled but not visible, 2: labeled and visible
    """
    # Convert input to numpy array if it's a list
    keypoints = np.array(keypoints)
    
    # Validate input shape
    if keypoints.shape != (17, 2):
        raise ValueError(f"Expected keypoints shape (17, 2), got {keypoints.shape}")
    
    # Handle visibility
    if visibility is None:
        visibility = np.full(17, 2)  # All keypoints visible
    else:
        visibility = np.array(visibility)
        if visibility.shape != (17,):
            raise ValueError(f"Expected visibility shape (17,), got {visibility.shape}")
    
    # Convert to torch tensors
    x = torch.tensor(keypoints, dtype=torch.float)
    v = torch.tensor(visibility, dtype=torch.float).view(-1, 1)
    
    # Create edge index for the pose graph
    edge_index = torch.tensor([
        [0, 1], [1, 0],  # nose - left eye
        [0, 2], [2, 0],  # nose - right eye
        [1, 3], [3, 1],  # left eye - left ear
        [2, 4], [4, 2],  # right eye - right ear
        [5, 6], [6, 5],  # left shoulder - right shoulder
        [5, 7], [7, 5],  # left shoulder - left elbow
        [6, 8], [8, 6],  # right shoulder - right elbow
        [7, 9], [9, 7],  # left elbow - left wrist
        [8, 10], [10, 8],  # right elbow - right wrist
        [5, 11], [11, 5],  # left shoulder - left hip
        [6, 12], [12, 6],  # right shoulder - right hip
        [11, 12], [12, 11],  # left hip - right hip
        [11, 13], [13, 11],  # left hip - left knee
        [12, 14], [14, 12],  # right hip - right knee
        [13, 15], [15, 13],  # left knee - left ankle
        [14, 16], [16, 14],  # right knee - right ankle
    ], dtype=torch.long)
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index.t().contiguous(), vis=v)
    return data

def predict_single(model, data, device):
    """Make prediction for a single pose."""
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        prob = torch.exp(output)  # Convert log probabilities to probabilities
    return pred.item(), prob[0].cpu().numpy()

def predict_pose(keypoints, visibility=None):
    """
    Predict whether a pose is correct or incorrect.
    
    Args:
        keypoints: numpy array or list of shape [17, 2] containing x,y coordinates in COCO format
        visibility: optional numpy array or list of shape [17] containing visibility flags
    
    Returns:
        prediction: "Correct" or "Incorrect"
        confidence: confidence score of the prediction
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the trained model
    model_path = 'models/model-1/best_model.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = load_trained_model(model_path, device)
    
    # Prepare input data
    data = prepare_single_pose_input(keypoints, visibility)
    
    # Make prediction
    prediction, probabilities = predict_single(model, data, device)
    
    # Convert prediction to class name and get confidence
    pred_class = "Correct" if prediction == 1 else "Incorrect"
    confidence = max(probabilities)
    
    return pred_class, confidence

def main():
    # Example usage
    print("Example of using the prediction script:")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Example 1: Test on sample dataset
    print("\n1. Testing on sample dataset:")
    
    # Load test dataset
    _, test_dataset = load_dataset(
        root_dir='data_v1',
        annotation_dir='annotations_v1'
    )
    
    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Load model
    model = load_trained_model('models/model-1/best_model.pth', device)
    
    # Make predictions
    model.eval()
    predictions = []
    probabilities = []
    labels = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            pred = output.max(dim=1)[1]
            prob = torch.exp(output)
            
            predictions.extend(pred.cpu().numpy())
            probabilities.extend(prob.cpu().numpy())
            labels.extend(data.y.cpu().numpy())
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=['Incorrect', 'Correct']))
    
    # Example 2: Single pose prediction
    print("\n2. Example of predicting a single pose:")
    
    # Example keypoints (replace with actual keypoints)
    example_keypoints = np.array([
        [0.5, 0.5],  # nose
        [0.45, 0.45],  # left_eye
        [0.55, 0.45],  # right_eye
        [0.4, 0.5],  # left_ear
        [0.6, 0.5],  # right_ear
        [0.3, 0.7],  # left_shoulder
        [0.7, 0.7],  # right_shoulder
        [0.2, 0.9],  # left_elbow
        [0.8, 0.9],  # right_elbow
        [0.1, 1.1],  # left_wrist
        [0.9, 1.1],  # right_wrist
        [0.4, 1.2],  # left_hip
        [0.6, 1.2],  # right_hip
        [0.35, 1.5],  # left_knee
        [0.65, 1.5],  # right_knee
        [0.3, 1.8],  # left_ankle
        [0.7, 1.8],  # right_ankle
    ])
    
    # Example visibility (all keypoints visible)
    example_visibility = np.ones(17) * 2
    
    # Make prediction
    pred_class, confidence = predict_pose(example_keypoints, example_visibility)
    
    print(f"\nPrediction: {pred_class}")
    print(f"Confidence: {confidence:.2%}")
    
    print("\nInput Format Requirements:")
    print("- Keypoints: array/list of shape (17, 2) containing x,y coordinates")
    print("- Visibility: optional array/list of shape (17) containing visibility flags")
    print("  0: not labeled, 1: labeled but not visible, 2: labeled and visible")

if __name__ == '__main__':
    main() 