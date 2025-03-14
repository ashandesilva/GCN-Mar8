import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split

from data.pose_image_dataset import PoseImageDataset
from models.cnn_models import CNNPoseClassifier
from utils.metrics import compute_metrics
from utils.visualization import plot_training_curves, plot_model_comparison
from configs.model_configs import MODEL_CONFIGS

def load_annotations(annotation_file):
    print(f"Loading annotations from {annotation_file}")
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    print(f"Loaded {len(annotations['annotations'])} annotations")
    return annotations

def parse_keypoints(annotations, label):
    data = []
    for ann in annotations['annotations']:
        keypoints = ann['keypoints']
        keypoints_array = np.array(keypoints).reshape(-1, 3)
        coords = keypoints_array[:, :2]
        visibility = keypoints_array[:, 2]
        data.append((coords, visibility, label))
    print(f"Parsed {len(data)} keypoints with label {label}")
    return data

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, model_name, save_dir):
    best_f1 = 0
    train_losses = []
    val_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation phase
        metrics = compute_metrics(model, val_loader, device)
        
        # Store metrics
        for metric, value in metrics.items():
            val_metrics[metric].append(value)
        
        # Print progress
        print(f'{model_name} - Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_loss:.4f}')
        print(f'Val Metrics:', metrics)
        
        # Save best model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
            }, os.path.join(save_dir, f'{model_name}_best.pth'))
    
    return {
        'train_losses': train_losses,
        'val_metrics': val_metrics
    }

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('results', f'benchmark_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Load annotations
    correct_annotations = load_annotations('../annotations_v3/n6-269-299-correct.json')
    incorrect_annotations = load_annotations('../annotations_v3/lumbar-neth-K-1.1-285-w53445.json')

    # Parse keypoints
    correct_data = parse_keypoints(correct_annotations, label=1)
    incorrect_data = parse_keypoints(incorrect_annotations, label=0)

    # Combine and split data
    all_data = correct_data + incorrect_data
    print(f"Total data points: {len(all_data)}")
    train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42)
    print(f"Training data points: {len(train_data)}, Validation data points: {len(val_data)}")
    
    results = {}
    
    # List of models to skip
    models_to_skip = [
        'vgg19', 'vgg16', 'DenseNet201', 'InceptionV3'
    ]

    for model_name, config in MODEL_CONFIGS.items():
        if model_name in models_to_skip:
            print(f"Skipping {model_name}")
            continue
        print(f"\nTraining {model_name}")
        
        # Create datasets and dataloaders
        train_dataset = PoseImageDataset(train_data, image_size=config['image_size'])
        val_dataset = PoseImageDataset(val_data, image_size=config['image_size'])
        
        train_loader = DataLoader(train_dataset, 
                                batch_size=config['batch_size'], 
                                shuffle=True)
        val_loader = DataLoader(val_dataset, 
                              batch_size=config['batch_size'], 
                              shuffle=False)
        
        # Create model and training components
        model = CNNPoseClassifier(model_name).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), 
                                   lr=config['learning_rate'])
        
        # Train model
        results[model_name] = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            config['epochs'], device, model_name, save_dir
        )
    
    # Save results
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f)
    
    # Plot results
    plot_training_curves(results, 
                        os.path.join(save_dir, 'training_curves.png'))
    plot_model_comparison(results, 
                         os.path.join(save_dir, 'model_comparison.png'))

if __name__ == '__main__':
    main()