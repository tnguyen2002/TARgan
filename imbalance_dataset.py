import json
import os
import random

def create_imbalanced_dataset(json_file, output_file, ratios):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Initialize dictionary to store imbalanced dataset
    imbalanced_data = {'labels': []}

    # Count occurrences of each class
    class_counts = {}
    for item in data['labels']:
        label = item[1]
        class_counts[label] = class_counts.get(label, 0) + 1

    # Calculate number of samples to remove for each class
    target_counts = {}
    for label, ratio in ratios.items():
        target_counts[label] = int(class_counts[label] * ratio)
    
    print(target_counts)

    # Randomly select images to remove from each class
    for item in data['labels']:
        path = item[0]
        label = item[1]

        if target_counts[label] > 0:
            # Add item to imbalanced dataset
            imbalanced_data['labels'].append([path, label])
            target_counts[label] -= 1

    # Write imbalanced dataset to JSON file
    with open(output_file, 'w') as f:
        json.dump(imbalanced_data, f, indent=4)


# Example usage
ratios = {
    0: 0.01, 
    1: 1, 
    2: 1,
    3: 0.01,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
    9: 1}  # Example ratios for classes 0, 1, and 2

create_imbalanced_dataset('dataset/cifar10/dataset.json', 'dataset/cifar10/imbalanced_dataset.json', ratios)
