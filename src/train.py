import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

from data_processing import load_dataset
from gcn_model import PoseGCN

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for data in tqdm(train_loader, desc='Training'):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
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

def plot_metrics(train_losses, val_metrics, save_path='training_metrics.png'):
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

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    train_dataset, val_dataset = load_dataset(
        root_dir='data_v1',
        annotation_dir='annotations_v1'
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize model
    model = PoseGCN(num_node_features=2).to(device)  # 2 features: x, y coordinates
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    num_epochs = 100
    best_val_acc = 0
    train_losses = []
    val_metrics = {
        'accuracy': [], 'precision': [], 'recall': [], 'f1': []
    }
    
    # Training loop
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
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Plot metrics
        plot_metrics(train_losses, val_metrics)

if __name__ == '__main__':
    main() 