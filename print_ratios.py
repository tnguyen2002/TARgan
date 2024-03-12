import os
import json 

def print_class_ratios(dataset_path):
    # Load dataset json
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Count samples for each class
    class_counts = {}
    total_samples = len(dataset['labels'])
    for _, label in dataset['labels']:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    # Print ratios
    print("Class Ratios:")
    for label, count in sorted(class_counts.items()):
        ratio = count / total_samples
        print(f"Class {label}: {count}/{total_samples} ({ratio:.2f})")

# Example usage
dataset_path = 'dataset/cifar10/imbalanced_dataset.json'
print_class_ratios(dataset_path)