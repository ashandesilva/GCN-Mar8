import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class PoseGCN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(PoseGCN, self).__init__()
        
        # GCN layers
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 256)
        
        # Fully connected layers
        self.fc1 = torch.nn.Linear(256, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 2)  # 2 classes: correct and incorrect
        
        # Dropout layer
        self.dropout = torch.nn.Dropout(0.5)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Apply GCN layers with ReLU activation
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        
        x = F.relu(self.conv3(x, edge_index))
        
        # Global pooling
        x = global_mean_pool(x, data.batch if hasattr(data, 'batch') else None)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1) 