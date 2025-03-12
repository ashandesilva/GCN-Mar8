import torch
import json
import os
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

from data.pose_image_dataset import PoseImageDataset
from models.cnn_models import CNNPoseClassifier
from utils.metrics import compute_metrics
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

def evaluate_model(model, test_loader, device):
    return compute_metrics(model, test_loader, device)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load annotations
    correct_annotations = load_annotations('../annotations_v3/n6-269-299-correct.json')
    incorrect_annotations = load_annotations('../annotations_v3/lumbar-neth-K-1.1-285-w53445.json')

    # Parse keypoints
    correct_data = parse_keypoints(correct_annotations, label=1)
    incorrect_data = parse_keypoints(incorrect_annotations, label=0)

    # Combine and split data
    all_data = correct_data + incorrect_data
    train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
    
    results = {}
    
    # Load the latest results directory
    results_dirs = [d for d in os.listdir('results') if d.startswith('benchmark_')]
    latest_dir = max(results_dirs)
    models_dir = os.path.join('results', latest_dir)
    
    for model_name, config in MODEL_CONFIGS.items():
        print(f"\nEvaluating {model_name}")
        
        # Create dataset and dataloader
        test_dataset = PoseImageDataset(test_data, image_size=config['image_size'])
        test_loader = DataLoader(test_dataset, 
                               batch_size=config['batch_size'], 
                               shuffle=False)
        
        # Load model
        model = CNNPoseClassifier(model_name).to(device)
        checkpoint = torch.load(os.path.join(models_dir, f'{model_name}_best.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        results[model_name] = evaluate_model(model, test_loader, device)
    
    # Save evaluation results
    with open(os.path.join(models_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f)
    
    # Print results
    print("\nEvaluation Results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    main()