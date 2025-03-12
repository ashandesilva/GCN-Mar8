import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms

class PoseImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, image_size=(224, 224), line_width=2):
        self.data = data
        self.image_size = image_size
        self.line_width = line_width
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.data)
    
    def draw_pose(self, keypoints):
        # Create a blank image
        image = Image.new('RGB', self.image_size, color='white')
        draw = ImageDraw.Draw(image)
        
        # Scale keypoints to image size
        kp = keypoints.copy()
        scale_x = self.image_size[0] / np.max(kp[:, 0])
        scale_y = self.image_size[1] / np.max(kp[:, 1])
        kp[:, 0] *= scale_x
        kp[:, 1] *= scale_y
        
        # Define connections for COCO keypoints
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Face
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
        ]
        
        # Draw connections
        for start_idx, end_idx in connections:
            start_point = tuple(kp[start_idx, :2])
            end_point = tuple(kp[end_idx, :2])
            draw.line([start_point, end_point], 
                     fill='black', width=self.line_width)
        
        # Draw keypoints
        for point in kp:
            x, y = point[:2]
            draw.ellipse([x-3, y-3, x+3, y+3], 
                        fill='red', outline='red')
            
        return image
    
    def __getitem__(self, idx):
        coords, visibility, label = self.data[idx]
        image = self.draw_pose(coords)
        image_tensor = self.transform(image)
        return image_tensor, torch.tensor(label, dtype=torch.long)