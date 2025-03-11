# Training Breakdown

__Best validation accuracy: 0.8019__


# Final epochs
```
Epoch 001:
Train Loss: 0.6588
Val Accuracy: 0.7075, Precision: 0.0000, Recall: 0.0000, F1: 0.0000
Saved best model with validation accuracy: 0.7075
c:\Users\ashan\Documents\MyFYP\Mar-2025\GCN-Mar8\venv\Lib\site-packages\sklearn\metrics\_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))

Epoch 002:
Train Loss: 0.6114
Val Accuracy: 0.7075, Precision: 0.0000, Recall: 0.0000, F1: 0.0000
c:\Users\ashan\Documents\MyFYP\Mar-2025\GCN-Mar8\venv\Lib\site-packages\sklearn\metrics\_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
  
Epoch 096:
Train Loss: 0.5324
Val Accuracy: 0.7642, Precision: 1.0000, Recall: 0.1935, F1: 0.3243

Epoch 097:
Train Loss: 0.5178
Val Accuracy: 0.7925, Precision: 1.0000, Recall: 0.2903, F1: 0.4500

Epoch 098:
Train Loss: 0.5339
Val Accuracy: 0.7925, Precision: 1.0000, Recall: 0.2903, F1: 0.4500

Epoch 099:
Train Loss: 0.5031
Val Accuracy: 0.8019, Precision: 1.0000, Recall: 0.3226, F1: 0.4878

Epoch 100:
Train Loss: 0.5371
Val Accuracy: 0.7925, Precision: 0.8000, Recall: 0.3871, F1: 0.5217
```

## Setup Training
```
Using device: cpu
Loading annotations from c:\Users\ashan\Documents\MyFYP\Mar-2025\GCN-Mar8\annotations_v2
Loaded 366 annotations from c:\Users\ashan\Documents\MyFYP\Mar-2025\GCN-Mar8\annotations_v2\t5-sherul-300-395-correct.json
Number of images: 579
Categories: ['human']
Successfully processed 366 keypoint annotations
Loaded 160 annotations from c:\Users\ashan\Documents\MyFYP\Mar-2025\GCN-Mar8\annotations_v2\lumbar-K-1.1-160.json
Number of images: 464
Categories: ['human']
Successfully processed 160 keypoint annotations
Loaded 366 correct poses and 160 incorrect poses
Total dataset size: 526 samples
Dataset loaded: 420 training samples, 106 validation samples
Model and optimizer initialized!
```
# Explanation of the metrics
```
Epoch 100:
Train Loss: 0.5371
Val Accuracy: 0.7925, Precision: 0.8000, Recall: 0.3871, F1: 0.5217
```

1. **Train Loss: 0.5371**
   - This is the average negative log-likelihood loss on the training data
   - A decreasing loss generally indicates the model is learning, but 0.5371 suggests there's still some uncertainty in the model's predictions

2. **Val Accuracy: 0.7925 (79.25%)**
   - This is the overall accuracy on the validation set
   - It means the model correctly classified about 79% of all poses (both correct and incorrect)
   - This is a decent overall accuracy, but looking at the other metrics reveals some important details

3. **Precision: 0.8000 (80%)**
   - Precision = True Positives / (True Positives + False Positives)
   - In your case, when the model predicts a pose is incorrect (class 1), it's right 80% of the time
   - This is good - when the model flags a pose as incorrect, it's usually right

4. **Recall: 0.3871 (38.71%)**
   - Recall = True Positives / (True Positives + False Negatives)
   - This is quite low - the model is only detecting about 39% of all the actual incorrect poses
   - This means the model is missing many incorrect poses, classifying them as correct when they're actually incorrect
   - This is a significant issue if the goal is to catch incorrect poses

5. **F1: 0.5217 (52.17%)**
   - F1 score is the harmonic mean of precision and recall: 2 * (precision * recall) / (precision + recall)
   - The low F1 score is mainly due to the low recall
   - This indicates an imbalance in the model's performance

The key insight here is that your model has a **high-precision, low-recall** pattern for incorrect poses, which means:
- It's conservative in flagging poses as incorrect
- When it does flag a pose as incorrect, it's usually right (80% precision)
- But it misses many incorrect poses (only 39% recall)
- This might be due to:
  1. Class imbalance in your training data
  2. The model being biased toward predicting "correct" poses
  3. Some incorrect poses being too subtle for the model to detect

To improve this, you might consider:
1. Balancing your training data between correct and incorrect poses
2. Using class weights in your loss function to give more importance to incorrect poses
3. Collecting more examples of incorrect poses
4. Using techniques like data augmentation to generate more incorrect pose examples
5. Adjusting the model's decision threshold to favor recall over precision if catching incorrect poses is more important
