import json
import os
import shutil

# Read JSON file
with open('data.json', 'r') as file:
    data = json.load(file)

# Create folders for each numerical label and move files
for label, file_index in data['labels']:
    folder_name = str(label)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_name = f"00000/img{file_index:08d}.png"  # Assuming the file structure follows the given pattern
    source_path = os.path.join('source_folder', file_name)  # Adjust 'source_folder' accordingly
    destination_path = os.path.join(folder_name, file_name)
    shutil.move(source_path, destination_path)
    print(f"Moved {file_name} to folder {folder_name}")
