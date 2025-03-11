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