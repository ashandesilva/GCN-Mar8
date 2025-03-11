# Training Breakdown

__Best validation accuracy: 0.7714__


# Epochs
```
Epoch 074:
Train Loss: 0.6103
Val Accuracy: 0.7429, Precision: 0.6875, Recall: 0.9167, F1: 0.7857

Epoch 098:
Train Loss: 0.5849
Val Accuracy: 0.7714, Precision: 0.7000, Recall: 0.9722, F1: 0.8140

Epoch 100:
Train Loss: 0.5894
Val Accuracy: 0.7571, Precision: 0.6863, Recall: 0.9722, F1: 0.8046

```

## Setup Training
```
Using device: cpu
Loading annotations from c:\Users\ashan\Documents\MyFYP\Mar-2025\GCN-Mar8\annotations_v2
Loaded 189 annotations from c:\Users\ashan\Documents\MyFYP\Mar-2025\GCN-Mar8\annotations_v2\t5-sherul-300-195-correct.json
Number of images: 579
Categories: ['human']
Successfully processed 189 keypoint annotations
Loaded 160 annotations from c:\Users\ashan\Documents\MyFYP\Mar-2025\GCN-Mar8\annotations_v2\lumbar-K-1.1-160.json
Number of images: 464
Categories: ['human']
Successfully processed 160 keypoint annotations
Loaded 189 correct poses and 160 incorrect poses
Total dataset size: 349 samples
Dataset loaded: 279 training samples, 70 validation samples
Model and optimizer initialized!
```

# Explanation of the metrics
```
Epoch 100:
Train Loss: 0.5894
Val Accuracy: 0.7571, Precision: 0.6863, Recall: 0.9722, F1: 0.8046
```

Much more balanced than before in model-3:

Dataset Distribution:
- Correct poses: 189 samples (54.2%)
- Incorrect poses: 160 samples (45.8%)
- Total: 349 samples
- Split: 279 training / 70 validation (80/20 split)

Your new metrics are significantly better balanced than before:

1. **Recall improved dramatically**: 
   - Old: 0.3871 (39%)
   - New: 0.9722 (97%)
   - This means your model is now catching almost all incorrect poses (97% of them!)

2. **Precision decreased slightly but still good**:
   - Old: 0.8000 (80%)
   - New: 0.6863 (69%)
   - This slight decrease is expected given the huge improvement in recall

3. **F1 Score improved significantly**:
   - Old: 0.5217 (52%)
   - New: 0.8046 (80%)
   - This shows much better overall balance between precision and recall

Looking at your training graphs:
1. The training loss (top graph) shows steady decrease over epochs, indicating good learning
2. The metrics graph (bottom) shows:
   - Very stable accuracy (green line) around 75-77%
   - High recall (blue line) consistently above 90% in later epochs
   - Balanced precision (red line) and F1-score (yellow line)

This is a much healthier model because:
1. The dataset is nearly balanced (189 vs 160 samples)
2. It has high recall, which means it rarely misses incorrect poses
3. While precision dropped a bit, it's an acceptable trade-off given the critical importance of not missing incorrect poses
4. The F1 score of 0.80 indicates good overall performance

---

# What Next?
<think>The user is asking for guidance on three strategies to improve their GCN model for pose classification. Let me provide detailed advice on each approach:

1. Improving precision without sacrificing recall
2. Experimenting with different model architectures
3. Adding data augmentation

For each, I'll provide concrete steps and code examples where appropriate.</think># Guidance on Improving Your GCN Pose Classification Model

Let me provide practical guidance on each of the three improvement strategies:

## 1. Improving Precision Without Sacrificing Recall

This is tricky but possible with these approaches:

### a) Adjust Class Weights

Add class weights to your loss function to fine-tune the balance between precision and recall:

```python
# Calculate class weights based on validation results
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Extract validation labels
val_labels = [data.y.item() for data in val_dataset]
class_weights = compute_class_weight('balanced', classes=np.unique(val_labels), y=val_labels)
class_weights = torch.FloatTensor(class_weights).to(device)

# Modify your loss function in the training loop
loss = torch.nn.functional.nll_loss(output, data.y, weight=class_weights)
```

### b) Confidence Thresholding

Instead of using the default 0.5 threshold for classification, you can tune it:

```python
def evaluate_with_threshold(model, loader, device, threshold=0.5):
    model.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            probs = torch.exp(output)  # Convert log_softmax to probabilities
            
            # Apply custom threshold to class 1 (incorrect pose)
            pred = (probs[:, 1] > threshold).long()
            
            predictions.extend(pred.cpu().numpy())
            labels.extend(data.y.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', pos_label=1
    )
    
    return accuracy, precision, recall, f1

# Test different thresholds
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
results = []

for t in thresholds:
    acc, prec, rec, f1 = evaluate_with_threshold(model, val_loader, device, threshold=t)
    results.append((t, acc, prec, rec, f1))
    print(f"Threshold {t:.1f}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
```

### c) Focal Loss

Implement Focal Loss, which puts more emphasis on hard-to-classify examples:

```python
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        log_probs = torch.nn.functional.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        # Get the log probability for the correct class
        targets = targets.view(-1, 1)
        log_p = log_probs.gather(1, targets).view(-1)
        p = probs.gather(1, targets).view(-1)
        
        # Calculate focal loss
        loss = -((1 - p) ** self.gamma) * log_p
        
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets.view(-1))
            loss = alpha_t * loss
            
        return loss.mean()

# Use in training loop
focal_loss = FocalLoss(gamma=2.0, alpha=torch.tensor([0.25, 0.75]).to(device))
loss = focal_loss(output, data.y)
```

## 2. Experimenting with Different Model Architectures

### a) Deeper GCN

Add more GCN layers and residual connections:

```python
class DeepPoseGCN(torch.nn.Module):
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

### b) GAT (Graph Attention Network)

Try attention-based graph convolutions:

```python
from torch_geometric.nn import GATConv

class PoseGAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64, num_classes=2, heads=8):
        super(PoseGAT, self).__init__()
        
        # Graph Attention layers
        self.conv1 = GATConv(num_node_features, hidden_channels//heads, heads=heads)
        self.conv2 = GATConv(hidden_channels, hidden_channels//heads, heads=heads)
        
        # Final classification layers
        self.linear = torch.nn.Linear(hidden_channels, num_classes)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First attention layer
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Second attention layer
        x = self.conv2(x, edge_index)
        
        # Pooling
        x = global_mean_pool(x, data.batch)
        
        # Classification
        x = self.linear(x)
        
        return F.log_softmax(x, dim=1)
```

### c) GraphSAGE

Use sampling-based convolutions:

```python
from torch_geometric.nn import SAGEConv

class PoseSAGE(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64, num_classes=2):
        super(PoseSAGE, self).__init__()
        
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, num_classes)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        
        x = global_mean_pool(x, data.batch)
        x = self.linear(x)
        
        return F.log_softmax(x, dim=1)
```

## 3. Adding Data Augmentation

Graph data augmentation is trickier but very effective:

### a) Node Feature Perturbation

Add small noise to keypoint positions:

```python
def augment_keypoints(data, noise_level=0.02):
    # Clone the data to avoid modifying the original
    augmented_data = Data(
        x=data.x.clone(),
        edge_index=data.edge_index.clone(),
        y=data.y.clone(),
        visibility=data.visibility.clone() if hasattr(data, 'visibility') else None
    )
    
    # Add random noise to node features (keypoint positions)
    noise = torch.randn_like(augmented_data.x) * noise_level
    augmented_data.x = augmented_data.x + noise
    
    return augmented_data

# Apply during training
augmented_batch = []
for data in batch:
    if np.random.random() < 0.5:  # 50% chance to apply augmentation
        augmented_batch.append(augment_keypoints(data))
    else:
        augmented_batch.append(data)
```

### b) Edge Masking

Randomly drop edges to simulate occlusions:

```python
def mask_edges(data, drop_prob=0.1):
    augmented_data = Data(
        x=data.x.clone(),
        edge_index=data.edge_index.clone(),
        y=data.y.clone(),
        visibility=data.visibility.clone() if hasattr(data, 'visibility') else None
    )
    
    # Create a mask to drop edges randomly
    edge_mask = torch.rand(augmented_data.edge_index.size(1)) > drop_prob
    augmented_data.edge_index = augmented_data.edge_index[:, edge_mask]
    
    return augmented_data
```

### c) Keypoint Visibility Augmentation

Simulate randomly invisible keypoints:

```python
def augment_visibility(data, invisible_prob=0.1):
    augmented_data = Data(
        x=data.x.clone(),
        edge_index=data.edge_index.clone(),
        y=data.y.clone(),
        visibility=data.visibility.clone() if hasattr(data, 'visibility') else None
    )
    
    if hasattr(augmented_data, 'visibility'):
        # Randomly set some keypoints as invisible
        mask = torch.rand(augmented_data.visibility.size()) < invisible_prob
        augmented_data.visibility[mask] = 0.0
        
        # You might want to modify the features of invisible keypoints
        invisible_nodes = torch.where(augmented_data.visibility == 0)[0]
        if len(invisible_nodes) > 0:
            # For invisible keypoints, you could set their position to the mean position
            # or introduce a special value
            mean_pos = augmented_data.x.mean(dim=0)
            augmented_data.x[invisible_nodes] = mean_pos
    
    return augmented_data
```

### d) Creating an Augmentation Pipeline

Implement a custom transform for PyTorch Geometric:

```python
class PoseAugmentation(object):
    def __init__(self, noise_level=0.02, drop_edge_prob=0.1, invisible_prob=0.1, p=0.5):
        self.noise_level = noise_level
        self.drop_edge_prob = drop_edge_prob
        self.invisible_prob = invisible_prob
        self.p = p  # Probability of applying augmentation
        
    def __call__(self, data):
        if np.random.random() > self.p:
            return data
            
        # Apply a sequence of augmentations
        if np.random.random() < 0.5:
            data = augment_keypoints(data, self.noise_level)
            
        if np.random.random() < 0.5:
            data = mask_edges(data, self.drop_edge_prob)
            
        if np.random.random() < 0.5:
            data = augment_visibility(data, self.invisible_prob)
            
        return data

# Use in dataset
train_dataset.transform = PoseAugmentation()
```

## Implementation Strategy

I recommend trying these improvements in this order:

1. First try data augmentation - it's often the easiest way to improve performance without changing the model
2. Then explore class weights or focal loss to improve the precision/recall balance
3. Finally experiment with more advanced model architectures

For rapid experimentation, implement a proper validation and logging system:

```python
def train_and_evaluate(model, train_loader, val_loader, optimizer, num_epochs=100, device='cpu'):
    best_f1 = 0
    train_losses = []
    val_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train(model, train_loader, optimizer, device)
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
```

Would you like me to help implement any of these improvements specifically?
