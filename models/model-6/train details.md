## Compare Model Performance

```
=== Model Comparison ===
Original PoseGCN - Best F1: 0.7342
DeepPoseGCN - Best F1: 0.8052

=== Best Precision, Recall, Accuracy Comparison ===
Original PoseGCN - Best Precision: 0.7667
DeepPoseGCN - Best Precision: 1.0000

Original PoseGCN - Best Recall: 0.7632
DeepPoseGCN - Best Recall: 0.8421

Original PoseGCN - Best Accuracy: 0.7000
DeepPoseGCN - Best Accuracy: 0.7857

=== Best Model Details (Based on F1 Score) ===
Original PoseGCN - Best epoch: 187
  F1: 0.7342
  Precision: 0.7073
  Recall: 0.7632
  Accuracy: 0.7000

DeepPoseGCN - Best epoch: 61
  F1: 0.8052
  Precision: 0.7949
  Recall: 0.8158
  Accuracy: 0.7857
```

### Even with 200 epoch
```
Deep - time took 1:37
Original - time took 1:16
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
Class weights: tensor([0.8885, 1.1434])
Setup complete!
```