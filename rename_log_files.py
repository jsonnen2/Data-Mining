import os
import re

# File written by ChatGPT

def rename_files_in_folder(folder_path):
    # Define the regex pattern to match only underscores between `F=` and `_T=`
    pattern = re.compile(r'T=log_loss')
    
    for filename in os.listdir(folder_path):
        # Ensure we are working only with files, not directories
        if os.path.isfile(os.path.join(folder_path, filename)):
            # Apply regex to replace underscore with hyphen only in the specific section
            new_filename = pattern.sub('T=log-loss', filename)
            
            # Rename file only if the new filename is different
            if new_filename != filename:
                os.rename(
                    os.path.join(folder_path, filename),
                    os.path.join(folder_path, new_filename)
                )
                print(f'Renamed: {filename} -> {new_filename}')

def remove_files_with_pattern(folder_path):

    pattern = re.compile(r'T=CRIT=random')
    for filename in os.listdir(folder_path):
        # Check if the file matches the pattern
        if pattern.search(filename):
            file_path = os.path.join(folder_path, filename)
            # Remove the file
            os.remove(file_path)
            print(f"Removed file: {file_path}")

# Usage
folder_path = './_log/_recommend_9_hopeful'  # Replace with the path to your folder
rename_files_in_folder(folder_path)
