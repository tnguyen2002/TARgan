import json
import os

def update_dataset_json(original_json_path, generated_images_dir, new_json_path):
    # Load the original dataset.json
    with open(original_json_path, 'r') as f:
        dataset = json.load(f)
    
    # Assume the dataset structure contains a key "labels" that maps to a list
    original_labels = dataset['labels']
    
    # Iterate through each label directory (0-9) in the generated_images_dir
    for label in range(10):  # For labels 0 through 9
        label_dir = os.path.join(generated_images_dir, str(label))
        for image_name in os.listdir(label_dir):
            # Construct the relative path as it should be saved in the json
            # Adjust the path format according to your requirements
            image_path = os.path.join('viewmaker_diffaugment_generated/', str(label), image_name)
            # Append the new image path and its label to the dataset
            original_labels.append([image_path, label])
    
    # Save the updated dataset structure to a new json file
    with open(new_json_path, 'w') as f:
        json.dump(dataset, f, indent=4)

# Example usage
original_json_path = '/home/adamchun/TARgan/dataset/cifar-10/dataset.json'
generated_images_dir = '/home/adamchun/TARgan/dataset/cifar-10_viewmaker_diffaugment/viewmaker_diffaugment_generated'
new_json_path = '/home/adamchun/TARgan/dataset/cifar-10_viewmaker_diffaugment/viewmaker_diffaugment_orig_dataset.json'
update_dataset_json(original_json_path, generated_images_dir, new_json_path)