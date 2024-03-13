import json
import csv
import random

def convert_json_to_csv_with_neg_index(input_json_path, output_csv_path):
    # Load the data from the JSON file
    with open(input_json_path, 'r') as json_file:
        data = json.load(json_file)

    dataset_size = len(data['labels'])
    
    # Open the output CSV file for writing
    with open(output_csv_path, 'w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)
        
        # Write the header row
        csv_writer.writerow(['image_path', 'label', 'split', 'neg_index'])
        
        # Iterate over the data and write each row to the CSV
        for index, item in enumerate(data['labels']):
            image_path, label = item
            # Generate a random neg_index that is different from the current index
            neg_index = index
            while neg_index == index:
                neg_index = random.randint(0, dataset_size - 1)
            # The split is always 'train'
            csv_writer.writerow([image_path, label, 'train', neg_index])

# Example usage
input_json_path = '/home/anhn/TARgan/viewmaker/data/cifar10_imbalanced/dataset.json'
output_csv_path = '/home/anhn/TARgan/viewmaker/data/cifar10_imbalanced/cifar10_imbalanced.csv'


convert_json_to_csv_with_neg_index(input_json_path, output_csv_path)
