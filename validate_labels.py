import json

def validate_labels(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Iterate through each item in the "labels" list
    for item in data.get('labels', []):
        # Check if the item is a list containing a string followed by an integer
        if not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str) or not isinstance(item[1], int):
            print("Invalid item:", item)
            # Modify the item to conform to the required structure
            # Example correction:
            # item = [str(item), 0]
            # You might want to adjust this based on your specific requirements

    # Write the modified data back to the JSON file if needed
    # with open(json_file, 'w') as f:
    #     json.dump(data, f, indent=4)

# Example usage
validate_labels('dataset/cifar10/dataset.json')