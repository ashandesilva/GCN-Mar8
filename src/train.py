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

def evaluate(model, loader, device):
    model.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            pred = output.max(dim=1)[1]
            
            predictions.extend(pred.cpu().numpy())
            labels.extend(data.y.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    return accuracy, precision, recall, f1

def plot_metrics(train_losses, val_metrics, save_path='models/training_metrics.png'):
    """Plot training metrics and save to file"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot training loss
    ax1.plot(train_losses, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot validation metrics
    epochs = range(len(val_metrics['accuracy']))
    ax2.plot(epochs, val_metrics['accuracy'], label='Accuracy')
    ax2.plot(epochs, val_metrics['precision'], label='Precision')
    ax2.plot(epochs, val_metrics['recall'], label='Recall')
    ax2.plot(epochs, val_metrics['f1'], label='F1-Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved training plot to {save_path}")

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