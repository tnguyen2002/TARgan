import json
import shutil
import os

def validate_labels_and_copy_images(json_file, source_folder, destination_folder):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Create destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Copy the JSON file to the destination folder
    shutil.copy(json_file, destination_folder)

    # Iterate through each item in the "labels" list
    for item in data.get('labels', []):
        if len(item) != 2 or not isinstance(item[0], str) or not isinstance(item[1], int):
            print("Invalid item:", item)
            continue

        image_path = os.path.join(source_folder, item[0])
        if os.path.exists(image_path):
            # Get the directory name from the original path
            image_dir = os.path.dirname(image_path)
            # Create the corresponding directory in the destination folder
            dest_dir = os.path.join(destination_folder, image_dir)
            os.makedirs(dest_dir, exist_ok=True)
            # Copy the image to the destination directory
            shutil.copy(image_path, dest_dir)
        else:
            print("Image not found:", image_path)

# Example usage
json_file = 'dataset/cifar10/dataset.json'  # Assuming you have created this JSON file
source_folder = 'dataset/cifar10'  # Assuming this is the folder containing the images
destination_folder = 'dataset/imbalanced_cifar10'  # Specify the destination folder
validate_labels_and_copy_images(json_file, source_folder, destination_folder)

# you have to mess around a bit with the folder structure to get it to match the original,
# this code kinda scuffed and doesnt get the folder structure to match exactly