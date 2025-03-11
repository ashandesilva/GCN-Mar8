## 500 epoch
Original - 3m 39sec
Deep - 1m 35sec

### Deep
```
Epoch 228:
Train Loss: 0.5158
Val Accuracy: 0.8000, Precision: 0.7407, Recall: 0.7407, F1: 0.7407
Current learning rate: 0.000250
EarlyStopping counter: 49 out of 50

Epoch 229:
Train Loss: 0.4698
Val Accuracy: 0.8286, Precision: 0.8000, Recall: 0.7407, F1: 0.7692
Current learning rate: 0.000250
EarlyStopping counter: 50 out of 50
Early stopping triggered at epoch 229
```

```
=== Model Comparison ===
Original PoseGCN - Best F1: 0.5763
DeepPoseGCN - Best F1: 0.8511

=== Best Precision, Recall, Accuracy Comparison ===
Original PoseGCN - Best Precision: 0.5312
DeepPoseGCN - Best Precision: 1.0000

Original PoseGCN - Best Recall: 0.6296
DeepPoseGCN - Best Recall: 0.8889

Original PoseGCN - Best Accuracy: 0.6429
DeepPoseGCN - Best Accuracy: 0.9000

=== Best Model Details (Based on F1 Score) ===
Original PoseGCN - Best epoch: 1
  F1: 0.5763
  Precision: 0.5312
  Recall: 0.6296
  Accuracy: 0.6429

DeepPoseGCN - Best epoch: 179
  F1: 0.8511
  Precision: 1.0000
  Recall: 0.7407
  Accuracy: 0.9000
  ```

### Analysis and Conclusions
Original model reached its best performance at epoch 1
Deep model reached its best performance at epoch 179

Learning rate was reduced 0 times for the original model
Learning rate was reduced 1 times for the deep model