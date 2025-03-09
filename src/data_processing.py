import json
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
import os

class PoseDataset(Dataset):
    def __init__(self, root_dir, annotation_dir, transform=None):
        super(PoseDataset, self).__init__()
        self.root_dir = root_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        
        # Load annotations
        print(f"Loading annotations from {annotation_dir}")
        self.correct_annotations = self._load_json(os.path.join(annotation_dir, "correct_n2_3_7_100.json"))
        self.incorrect_annotations = self._load_json(os.path.join(annotation_dir, "lumbar_3_7_dell_111.json"))
        
        print(f"Loaded {len(self.correct_annotations)} correct poses and {len(self.incorrect_annotations)} incorrect poses")
        
        # Combine and create labels
        self.all_data = []
        self.all_labels = []
        self.all_visibility = []  # New list to store visibility flags
        
        # Add correct poses (label 0)
        for i, pose in enumerate(self.correct_annotations):
            try:
                # Debug print for the first pose
                if i == 0:
                    print(f"Sample correct pose data: {pose[:2]}")  # Print first 2 keypoints
                
                # Convert string coordinates to float arrays and extract visibility
                keypoints = []
                visibility = []
                for point in pose:
                    try:
                        if isinstance(point, str):
                            # Try comma split first
                            try:
                                values = [float(x.strip()) for x in point.split(',')]
                            except ValueError:
                                # If comma split fails, try space split
                                values = [float(x.strip()) for x in point.split()]
                        elif isinstance(point, (list, tuple)):
                            values = [float(x) for x in point]
                        else:
                            raise ValueError(f"Unexpected point format: {type(point)}")
                        
                        if len(values) == 3:  # COCO format [x, y, v]
                            coords = values[:2]
                            vis = int(values[2])  # visibility flag
                        elif len(values) == 2:  # Just coordinates
                            coords = values
                            vis = 2  # Assume visible if not specified
                        else:
                            raise ValueError(f"Expected 2 or 3 values, got {len(values)}")
                            
                        keypoints.append(coords)
                        visibility.append(vis)
                    except Exception as e:
                        print(f"Error processing point in correct pose {i}: {point}")
                        print(f"Error: {str(e)}")
                        raise
                
                if len(keypoints) != 17:
                    raise ValueError(f"Expected 17 keypoints, got {len(keypoints)}")
                
                self.all_data.append(keypoints)
                self.all_labels.append(0)
                self.all_visibility.append(visibility)
            except Exception as e:
                print(f"Error processing correct pose {i}")
                print(f"Error: {str(e)}")
                raise
            
        # Add incorrect poses (label 1)
        for i, pose in enumerate(self.incorrect_annotations):
            try:
                # Debug print for the first pose
                if i == 0:
                    print(f"Sample incorrect pose data: {pose[:2]}")  # Print first 2 keypoints
                
                # Convert string coordinates to float arrays and extract visibility
                keypoints = []
                visibility = []
                for point in pose:
                    try:
                        if isinstance(point, str):
                            # Try comma split first
                            try:
                                values = [float(x.strip()) for x in point.split(',')]
                            except ValueError:
                                # If comma split fails, try space split
                                values = [float(x.strip()) for x in point.split()]
                        elif isinstance(point, (list, tuple)):
                            values = [float(x) for x in point]
                        else:
                            raise ValueError(f"Unexpected point format: {type(point)}")
                        
                        if len(values) == 3:  # COCO format [x, y, v]
                            coords = values[:2]
                            vis = int(values[2])  # visibility flag
                        elif len(values) == 2:  # Just coordinates
                            coords = values
                            vis = 2  # Assume visible if not specified
                        else:
                            raise ValueError(f"Expected 2 or 3 values, got {len(values)}")
                            
                        keypoints.append(coords)
                        visibility.append(vis)
                    except Exception as e:
                        print(f"Error processing point in incorrect pose {i}: {point}")
                        print(f"Error: {str(e)}")
                        raise
                
                if len(keypoints) != 17:
                    raise ValueError(f"Expected 17 keypoints, got {len(keypoints)}")
                
                self.all_data.append(keypoints)
                self.all_labels.append(1)
                self.all_visibility.append(visibility)
            except Exception as e:
                print(f"Error processing incorrect pose {i}")
                print(f"Error: {str(e)}")
                raise
    
    def _load_json(self, path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                print(f"Successfully loaded {path}")
                if len(data) > 0:
                    print(f"First item type: {type(data[0])}")
                    print(f"First item length: {len(data[0])}")
                    print(f"First item first element: {data[0][0]}")
                return data
        except Exception as e:
            print(f"Error loading {path}")
            print(f"Error: {str(e)}")
            raise
    
    def _create_edge_index(self):
        # Define the skeleton connections
        edges = [
            # Torso
            [0, 1], [1, 2], [2, 3],  # Spine
            [2, 4], [4, 5],          # Right arm
            [2, 6], [6, 7],          # Left arm
            [3, 8], [3, 9],          # Hips
            [8, 10], [10, 11],       # Right leg
            [9, 12], [12, 13],       # Left leg
            [4, 14], [6, 15],        # Shoulders
            [1, 16]                  # Neck
        ]
        
        # Create bidirectional edges
        edge_index = []
        for edge in edges:
            edge_index.extend([[edge[0], edge[1]], [edge[1], edge[0]]])
        
        return torch.tensor(edge_index, dtype=torch.long).t()
    
    def len(self):
        return len(self.all_data)
    
    def get(self, idx):
        pose_data = self.all_data[idx]
        label = self.all_labels[idx]
        visibility = self.all_visibility[idx]
        
        # Convert keypoints and visibility to tensors
        keypoints = torch.tensor(pose_data, dtype=torch.float)
        visibility = torch.tensor(visibility, dtype=torch.float).view(-1, 1)
        
        # Normalize keypoints
        keypoints = self._normalize_keypoints(keypoints)
        
        # Create edge index
        edge_index = self._create_edge_index()
        
        # Create PyG Data object with visibility information
        data = Data(
            x=keypoints,                    # Node features
            edge_index=edge_index,          # Edge connections
            y=torch.tensor([label]),        # Label
            visibility=visibility,          # Keypoint visibility
        )
        
        if self.transform:
            data = self.transform(data)
        
        return data
    
    def _normalize_keypoints(self, keypoints):
        """Normalize keypoints to have zero mean and unit variance"""
        mean = torch.mean(keypoints, dim=0)
        std = torch.std(keypoints, dim=0)
        return (keypoints - mean) / (std + 1e-7)  # Add small epsilon to avoid division by zero

def load_dataset(root_dir, annotation_dir):
    """Helper function to load the dataset"""
    dataset = PoseDataset(root_dir, annotation_dir)
    
    # Split into train/val sets (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    return train_dataset, val_dataset 