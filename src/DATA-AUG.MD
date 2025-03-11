

From the outline information :
- `train.py` contains training and evaluation functions
- `gcn_model.py` defines their GCN model
- `data_processing.py` handles dataset loading and processing
- `utils.py` has utility functions for visualization and prediction



Improve  pose classification GCN model through both architecture improvements and data augmentation.

## 1. Implement a Deeper GCN Architecture

Let's enhance your model in `gcn_model.py`:

```python:src/gcn_model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class PoseGCN(torch.nn.Module):
    """Original GCN model for pose classification."""
    def __init__(self, num_node_features):
        super(PoseGCN, self).__init__()
        # GCN layers
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        
        # Classification layers
        self.linear = torch.nn.Linear(64, 2)  # 2 classes: correct/incorrect
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        
        # Global pooling (average all node features)
        x = global_mean_pool(x, data.batch)
        
        # Classification layer
        x = self.linear(x)
        
        return F.log_softmax(x, dim=1)


class DeepPoseGCN(torch.nn.Module):
    """Enhanced deeper GCN model with residual connections and batch normalization."""
    def __init__(self, num_node_features, hidden_channels=64, num_classes=2):
        super(DeepPoseGCN, self).__init__()
        
        # Multiple GCN layers with residual connections
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        
        # Batch normalization layers
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        
        # Final classification layers
        self.linear1 = torch.nn.Linear(hidden_channels, hidden_channels//2)
        self.linear2 = torch.nn.Linear(hidden_channels//2, num_classes)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First block
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        
        # Second block with residual connection
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x2 = x2 + x1  # Residual connection
        
        # Third block with residual connection
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, p=0.2, training=self.training)
        x3 = x3 + x2  # Residual connection
        
        # Fourth block
        x4 = self.conv4(x3, edge_index)
        
        # Global pooling
        x = global_mean_pool(x4, data.batch)
        
        # Classification layers
        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        
        return F.log_softmax(x, dim=1)
```
## 2. Now let's add data augmentation to the `data_processing.py`:

```python:src/data_processing.py
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
// ... existing code ...
```
## 3. Finally, let's update the training script to use our new model and augmentation:

```python:src/train.py
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import os
from sklearn.utils.class_weight import compute_class_weight

from data_processing import load_dataset, PoseAugmentation
from gcn_model import PoseGCN, DeepPoseGCN

def train(model, train_loader, optimizer, device, class_weights=None):
    model.train()
    total_loss = 0
    
    for data in tqdm(train_loader, desc='Training'):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Use weighted loss if class weights are provided
        if class_weights is not None:
            loss = torch.nn.functional.nll_loss(output, data.y, weight=class_weights)
        else:
            loss = torch.nn.functional.nll_loss(output, data.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# ... keep evaluate and plot_metrics functions unchanged ...

def train_and_evaluate(model, train_loader, val_loader, optimizer, num_epochs=100, device='cpu', class_weights=None):
    best_f1 = 0
    train_losses = []
    val_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train(model, train_loader, optimizer, device, class_weights)
        train_losses.append(train_loss)
        
        # Evaluate
        val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, device)
        
        # Store metrics
        val_metrics['accuracy'].append(val_acc)
        val_metrics['precision'].append(val_prec)
        val_metrics['recall'].append(val_rec)
        val_metrics['f1'].append(val_f1)
        
        # Print progress
        print(f'Epoch {epoch+1:03d}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, '
              f'Recall: {val_rec:.4f}, F1: {val_f1:.4f}')
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'train_losses': train_losses
            }, f'models/model_{model.__class__.__name__}_best.pth')
            print(f"Saved best model with F1: {val_f1:.4f}")
            
    return train_losses, val_metrics

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    data_dir = 'data_v2'
    annotation_dir = 'annotations_v2'
    
    # Create augmentation transform
    augmentation = PoseAugmentation(
        noise_level=0.02,  # 2% noise relative to the normalized keypoints
        drop_edge_prob=0.1,  # 10% probability to drop an edge
        invisible_prob=0.1,  # 10% probability to mark a keypoint as invisible
        p=0.5  # 50% probability to apply augmentation to a sample
    )
    
    # Load datasets with augmentation
    train_dataset, val_dataset = load_dataset(
        root_dir=data_dir,
        annotation_dir=annotation_dir
    )
    
    # Apply augmentation to training dataset
    train_dataset.transform = augmentation
    
    # Compute class weights for balanced training
    train_labels = [data.y.item() for data in train_dataset]
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights: {class_weights}")
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Initialize and train the original model for comparison
    original_model = PoseGCN(num_node_features=2).to(device)
    original_optimizer = optim.Adam(original_model.parameters(), lr=0.001)
    
    print("\n=== Training Original PoseGCN ===")
    original_losses, original_metrics = train_and_evaluate(
        original_model, train_loader, val_loader, original_optimizer, 
        num_epochs=100, device=device, class_weights=class_weights
    )
    
    # Plot metrics for original model
    plot_metrics(original_losses, original_metrics, save_path='models/original_model_metrics.png')
    
    # Initialize and train the deep model
    deep_model = DeepPoseGCN(num_node_features=2).to(device)
    deep_optimizer = optim.Adam(deep_model.parameters(), lr=0.001, weight_decay=1e-4)  # Added weight decay for regularization
    
    print("\n=== Training DeepPoseGCN ===")
    deep_losses, deep_metrics = train_and_evaluate(
        deep_model, train_loader, val_loader, deep_optimizer, 
        num_epochs=100, device=device, class_weights=class_weights
    )
    
    # Plot metrics for deep model
    plot_metrics(deep_losses, deep_metrics, save_path='models/deep_model_metrics.png')
    
    # Compare the best performance of both models
    print("\n=== Model Comparison ===")
    print(f"Original PoseGCN - Best F1: {max(original_metrics['f1']):.4f}")
    print(f"DeepPoseGCN - Best F1: {max(deep_metrics['f1']):.4f}")

if __name__ == '__main__':
    main()
```
## How to Run Your Improved GCN Pipeline

I've implemented three key improvements to make your GCN more accurate:

1. **Deeper Model Architecture**:
   - Added `DeepPoseGCN` with 4 GCN layers (vs. 2 in original)
   - Added residual connections to help with gradient flow
   - Added batch normalization for better stability
   - Used hierarchical classification layers

2. **Data Augmentation**:
   - `PoseAugmentation` class implements three strategies:
     - Keypoint position noise (simulates slight variations in pose)
     - Edge masking (simulates occlusion)
     - Visibility augmentation (makes model robust to missing keypoints)

3. **Class Weighting**:
   - Added automatic class weight calculation for balanced training
   - Modified loss function to use these weights

### To Run the Improved Pipeline:

```bash
python src/train.py
```

The script will:
1. Train the original model with augmentation and class weights
2. Train the deeper model with the same setup
3. Compare their performance using F1 score
4. Save both models and their metrics plots

### What to Expect:

1. **Improved Precision**: Class weights help balance precision and recall
2. **Better Generalization**: Data augmentation makes the model more robust
3. **Faster Convergence**: Residual connections and batch normalization help with training
4. **Higher F1 Score**: The deep model should outperform the original model

### Additional Improvements to Consider:

1. **Learning Rate Scheduling**: Implement a learning rate scheduler to reduce learning rate over time:
   ```python
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
   # Then after validation in the training loop:
   scheduler.step(val_f1)
   ```

2. **Ensemble Models**: Train multiple models with different initializations and ensemble their predictions:
   ```python
   # For prediction with ensemble
   def ensemble_predict(models, data, device):
       probs_sum = None
       for model in models:
           model.eval()
           with torch.no_grad():
               output = model(data.to(device))
               probs = torch.exp(output)
               if probs_sum is None:
                   probs_sum = probs
               else:
                   probs_sum += probs
       return probs_sum / len(models)
   ```

3. **Cross-Validation**: Implement k-fold cross-validation for more robust evaluation:
   ```python
   from sklearn.model_selection import KFold
   
   kfold = KFold(n_splits=5, shuffle=True)
   for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
       # Train and evaluate model on this fold
   ```
