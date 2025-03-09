import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data

def visualize_pose(keypoints, prediction=None, save_path=None):
    """
    Visualize a pose with its keypoints and connections
    Args:
        keypoints: numpy array of shape (17, 2) containing x,y coordinates
        prediction: optional prediction label (0: correct, 1: incorrect)
        save_path: path to save the visualization
    """
    # Define connections between keypoints
    connections = [
        (0, 1), (1, 2), (2, 3),  # Spine
        (2, 4), (4, 5),          # Right arm
        (2, 6), (6, 7),          # Left arm
        (3, 8), (3, 9),          # Hips
        (8, 10), (10, 11),       # Right leg
        (9, 12), (12, 13),       # Left leg
        (4, 14), (6, 15),        # Shoulders
        (1, 16)                  # Neck
    ]
    
    plt.figure(figsize=(8, 8))
    
    # Plot connections
    for connection in connections:
        plt.plot([keypoints[connection[0]][0], keypoints[connection[1]][0]],
                 [keypoints[connection[0]][1], keypoints[connection[1]][1]], 'b-')
    
    # Plot keypoints
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c='r', s=50)
    
    if prediction is not None:
        plt.title(f'Prediction: {"Correct" if prediction == 0 else "Incorrect"}')
    
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def predict_single_pose(model, keypoints, device):
    """
    Make prediction for a single pose
    Args:
        model: trained GCN model
        keypoints: numpy array of shape (17, 2) containing x,y coordinates
        device: torch device
    Returns:
        prediction: 0 (correct) or 1 (incorrect)
        confidence: prediction confidence
    """
    model.eval()
    
    # Convert keypoints to torch tensor
    x = torch.tensor(keypoints, dtype=torch.float)
    
    # Create edge index (same as in training)
    edges = [
        [0, 1], [1, 2], [2, 3],  # Spine
        [2, 4], [4, 5],          # Right arm
        [2, 6], [6, 7],          # Left arm
        [3, 8], [3, 9],          # Hips
        [8, 10], [10, 11],       # Right leg
        [9, 12], [12, 13],       # Left leg
        [4, 14], [6, 15],        # Shoulders
        [1, 16]                  # Neck
    ]
    
    edge_index = []
    for edge in edges:
        edge_index.extend([[edge[0], edge[1]], [edge[1], edge[0]]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index)
    data = data.to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(data)
        probabilities = torch.exp(output)
        prediction = output.max(dim=1)[1].item()
        confidence = probabilities[0][prediction].item()
    
    return prediction, confidence

def normalize_keypoints(keypoints):
    """
    Normalize keypoints to have zero mean and unit variance
    Args:
        keypoints: numpy array of shape (17, 2)
    Returns:
        normalized keypoints
    """
    mean = np.mean(keypoints, axis=0)
    std = np.std(keypoints, axis=0)
    return (keypoints - mean) / std 