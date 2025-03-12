import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_training_curves(results, save_path):
    plt.figure(figsize=(20, 15))
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        
        for model_name, data in results.items():
            plt.plot(data['val_metrics'][metric], 
                    label=model_name, alpha=0.7)
        
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} Comparison')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_model_comparison(results, save_path):
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Prepare data for plotting
    data = {metric: [max(results[model]['val_metrics'][metric]) 
                    for model in models] 
            for metric in metrics}
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, data[metric], width, label=metric)
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Best Performance Comparison')
    plt.xticks(x + width*1.5, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()