''' fish4knowledge_for_yolo.py
> This script is used to change the class labels of the fish4knowledge dataset to YOLO format.
'''

from tqdm import tqdm
import os

# Set the working directory
os.chdir('data/fish4knowledge') if os.getcwd().split('\\')[-1] != 'fish4knowledge' else None

# Change the class labels of the fish4knowledge dataset to YOLO format
dirs = ['train', 'test', 'valid']

for d in dirs:
    for f in tqdm(os.listdir(f"{d}/labels"), desc=f'Changing class labels for {d} directory...', unit='files', leave=False):
        # Read the lines of the file
        with open(f'{d}/labels/{f}', 'r+') as file:
            lines = file.readlines()
        # Change the class_id to 0
        lines = ['0' + line[1:] for line in lines]
        # Rewrite the lines to the file
        with open(f'{d}/labels/{f}', 'w') as file:
            file.writelines(lines)