import os
import re
from pathlib import Path
import cv2
from tqdm import tqdm

def make_unique_file_name(directory_path, base_name):
    # Initialize a counter to add to the base name if needed
    counter = 1
    new_name = base_name
    
    while os.path.exists(os.path.join(directory_path, new_name)):
        # If a file with the new name already exists, increment the counter
        new_name = f"{base_name}_{counter}"
        counter += 1
    
    return new_name

def keep_allowed_characters(directory_path):
    # Define a regular expression pattern to match characters to be kept
    pattern = r'[A-Za-z0-9-_.]'
    
    # Iterate over the files in the specified directory
    for filename in os.listdir(directory_path):
        # Construct the full file path
        file_path = os.path.join(directory_path, filename)
        
        # Check if the file path is a file (not a directory)
        if os.path.isfile(file_path):
            # Extract characters that match the pattern
            new_filename = ''.join(re.findall(pattern, filename))
            
            # Create a unique file name if necessary
            if new_filename != filename:
                new_filename = make_unique_file_name(directory_path, new_filename)
                new_file_path = os.path.join(directory_path, new_filename)
                os.rename(file_path, new_file_path)
                print(f'Renamed: {filename} -> {new_filename}')
            else:
                print(f'No changes needed for: {filename}')

# Replace 'your_directory_path' with the actual directory path you want to process
directory_path = Path(r"C:\Coding projects\yolo\data\ready_to_go_data\train_test_split\val").as_posix()

keep_allowed_characters(directory_path)

# for image_path in tqdm(directory_path.rglob('*.png'), total=len(list(directory_path.rglob('*.png')))):
#     img = cv2.imread(image_path.as_posix())
#     if img is None:
#         print('a')