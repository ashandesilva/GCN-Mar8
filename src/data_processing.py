import json
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
import os

class PoseAugmentation(object):
    """Data augmentation for pose keypoints in graph format."""
    def __init__(self, noise_level=0.02, drop_edge_prob=0.1, invisible_prob=0.1, p=0.5):
        self.noise_level = noise_level
        self.drop_edge_prob = drop_edge_prob
        self.invisible_prob = invisible_prob
        self.p = p  # Probability of applying augmentation
        
    def __call__(self, data):
        if np.random.random() > self.p:
            return data
            
        # Clone the data to avoid modifying the original
        augmented_data = Data(
            x=data.x.clone(),
            edge_index=data.edge_index.clone(),
            y=data.y.clone(),
            visibility=data.visibility.clone() if hasattr(data, 'visibility') else None,
            batch=data.batch if hasattr(data, 'batch') else None
        )
        
        # Apply a sequence of augmentations
        augmented_data = self._augment_keypoints(augmented_data)
        augmented_data = self._mask_edges(augmented_data)
        
        if hasattr(augmented_data, 'visibility'):
            augmented_data = self._augment_visibility(augmented_data)
            
        return augmented_data
        
    def _augment_keypoints(self, data, noise_level=None):
        """Add random noise to keypoint positions."""
        if np.random.random() > 0.5:
            return data
            
        if noise_level is None:
            noise_level = self.noise_level
            
        # Add random noise to node features (keypoint positions)
        noise = torch.randn_like(data.x) * noise_level
        data.x = data.x + noise
        
        return data
        
    def _mask_edges(self, data, drop_prob=None):
        """Randomly drop edges to simulate occlusions."""
        if np.random.random() > 0.5:
            return data
            
        if drop_prob is None:
            drop_prob = self.drop_edge_prob
        
        # Create a mask to drop edges randomly
        edge_mask = torch.rand(data.edge_index.size(1)) > drop_prob
        data.edge_index = data.edge_index[:, edge_mask]
        
        return data
        
    def _augment_visibility(self, data, invisible_prob=None):
        """Simulate randomly invisible keypoints."""
        if np.random.random() > 0.5:
            return data
            
        if invisible_prob is None:
            invisible_prob = self.invisible_prob
        
        # Randomly set some keypoints as invisible
        mask = torch.rand(data.visibility.size()) < invisible_prob
        data.visibility[mask] = 0.0
        
        # For invisible keypoints, you could set their position to mean
        # or introduce some noise to make the model more robust
        invisible_nodes = torch.where(data.visibility == 0)[0]
        if len(invisible_nodes) > 0:
            mean_pos = data.x.mean(dim=0)
            # Add some noise to the mean position
            noise = torch.randn_like(mean_pos) * 0.1
            data.x[invisible_nodes] = mean_pos + noise
        
        return data

class PoseDataset(Dataset):
    def __init__(self, root_dir, annotation_dir, transform=None):
        super(PoseDataset, self).__init__()
        self.root_dir = root_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.num_keypoints = 17  # COCO format has 17 keypoints
        
        # Load annotations
        print(f"Loading annotations from {annotation_dir}")
        # self.correct_annotations = self._load_json(os.path.join(annotation_dir, "correct_n2_3_7_100.json"))
        # self.incorrect_annotations = self._load_json(os.path.join(annotation_dir, "lumbar_3_7_dell_111.json"))
        self.correct_annotations = self._load_json(os.path.join(annotation_dir, "t5-sherul-300-195-correct.json"))
        self.incorrect_annotations = self._load_json(os.path.join(annotation_dir, "lumbar-K-1.1-160.json"))
        
        print(f"Loaded {len(self.correct_annotations)} correct poses and {len(self.incorrect_annotations)} incorrect poses")
        
        # Combine and create labels
        self.all_data = []
        self.all_labels = []
        self.all_visibility = []
        
        # Process correct poses (label 0)
        for keypoints in self.correct_annotations:
            self.all_data.append(keypoints[:, :2])  # Only x,y coordinates
            self.all_visibility.append(keypoints[:, 2])  # Visibility values
            self.all_labels.append(0)
            
        # Process incorrect poses (label 1)
        for keypoints in self.incorrect_annotations:
            self.all_data.append(keypoints[:, :2])  # Only x,y coordinates
            self.all_visibility.append(keypoints[:, 2])  # Visibility values
            self.all_labels.append(1)
            
        print(f"Total dataset size: {len(self.all_data)} samples")
        
    def _load_json(self, json_path):
        """
        Load and parse a COCO format JSON file containing keypoint annotations.
        
        Args:
            json_path (str): Path to the JSON file
            
        Returns:
            list: List of keypoint annotations
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            # Validate COCO format structure
            required_keys = ['images', 'annotations', 'categories']
            if not all(key in data for key in required_keys):
                raise ValueError(f"JSON file missing required COCO format keys. Required: {required_keys}")
                
            print(f"Loaded {len(data['annotations'])} annotations from {json_path}")
            print(f"Number of images: {len(data['images'])}")
            print(f"Categories: {[cat['name'] for cat in data['categories']]}")
            
            # Extract keypoint annotations
            keypoint_annotations = []
            for ann in data['annotations']:
                if 'keypoints' not in ann:
                    continue
                    
                keypoints = ann['keypoints']
                # COCO format has [x, y, v] triplets
                if len(keypoints) != self.num_keypoints * 3:
                    print(f"Warning: Expected {self.num_keypoints * 3} values for keypoints, got {len(keypoints)}")
                    continue
                    
                # Reshape into [N, 3] format where N is number of keypoints
                keypoints = np.array(keypoints).reshape(-1, 3)
                keypoint_annotations.append(keypoints)
                
            if not keypoint_annotations:
                raise ValueError(f"No valid keypoint annotations found in {json_path}")
                
            print(f"Successfully processed {len(keypoint_annotations)} keypoint annotations")
            return keypoint_annotations
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON file {json_path}: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error loading annotations from {json_path}: {str(e)}")
    
    def _create_edge_index(self):
        """
        Create edge connections for COCO keypoints.
        COCO keypoint order: [nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder,
                            left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip,
                            left_knee, right_knee, left_ankle, right_ankle]
        """
        edges = [
            # Face
            [0, 1], [0, 2],  # Nose to eyes
            [1, 3], [2, 4],  # Eyes to ears
            
            # Arms
            [5, 7], [7, 9],    # Left arm
            [6, 8], [8, 10],   # Right arm
            
            # Torso
            [5, 6],    # Shoulders
            [5, 11], [6, 12],  # Shoulders to hips
            [11, 12],  # Hips
            
            # Legs
            [11, 13], [13, 15],  # Left leg
            [12, 14], [14, 16],  # Right leg
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