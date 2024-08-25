''' deepfish_for_yolo.py 
> This script is used to convert the deepfish dataset to yolo format.
'''

from typing import Literal
from tqdm import tqdm
import cv2 as cv
import json, logging, os, shutil, random

# Set the working directory
if os.getcwd().split('\\')[-1] != 'ozfish':
    os.chdir('data/ozfish')

# Organize the image directories for YOLO format
def organize_image_directories():
    ''' Organize the image directories for YOLO format. '''
    
    # Define three directories: train, test and val
    dirs = ['train', 'test', 'valid']
    os.makedirs('train', exist_ok=True)
    os.makedirs('test', exist_ok=True)
    os.makedirs('valid', exist_ok=True)

    # Define images and labels directories for each of the three directories
    sub_dirs = ['images', 'labels']
    for d in dirs:
        for sd in sub_dirs:
            os.makedirs(f'{d}/{sd}', exist_ok=True)

    # Set the seed for reproducibility 
    random.seed(42)

    def move_images(train_files : list, batch : str = "batch01", dest : Literal["train", "valid"] = "train"):
        for f in tqdm(train_files, desc=f'Moving {batch} images to {dest} ...', unit='files'):
            shutil.move(f'{batch}/{f}', f'{dest}/images/{f}')

    # Divide the batch01 into train and valid
    train_batch01 = random.sample(os.listdir('batch01'), int(len(os.listdir('batch01')) * 0.8))
    valid_batch01 = [x for x in os.listdir('batch01') if x not in train_batch01]

    move_images(train_batch01, "batch01", "train")
    move_images(valid_batch01, "batch01", "valid")

    # Divide the batch02 into train and valid
    train_batch02 = random.sample(os.listdir('batch02'), int(len(os.listdir('batch02')) * 0.8))
    valid_batch02 = [x for x in os.listdir('batch02') if x not in train_batch02]

    move_images(train_batch02, "batch02", "train")
    move_images(valid_batch02, "batch02", "valid")

    # Divide the batch03 into train and valid
    train_batch03 = random.sample(os.listdir('batch03'), int(len(os.listdir('batch03')) * 0.8))
    valid_batch03 = [x for x in os.listdir('batch03') if x not in train_batch03]

    move_images(train_batch03, "batch03", "train")
    move_images(valid_batch03, "batch03", "valid")

    os.rmdir('batch01')
    os.rmdir('batch02')
    os.rmdir('batch03')

# Organize the label directories for YOLO format
def organize_labels_directories():
    ''' Organize the label directories for YOLO format. '''
    
    def load_multiple_json_objects(filename: str = 'batch01') -> list:
        with open(filename, 'r') as f:
            objects = []
            for line in f:
                try:
                    objects.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logging.error(f'Error in loading the JSON object: {e}')
        return objects
    
    def write_labels_to_file(data: list):
        for item in tqdm(data, desc='Writing labels to file ...', unit='files'):
            # Get the filename
            filename = os.path.join("train/labels", item['source-ref']) if item['source-ref'] in os.listdir('train/images') else os.path.join("valid/labels", item['source-ref'])
            filename = filename.rsplit('.', 1)[0]
            # Get the batch name
            batch_name = list(item.keys())[1]
            # Get the annotations
            annotations = item[batch_name]['annotations']
            # Get the image size (width x height)
            image_width, image_height = item[batch_name]['image_size'][0]['width'], item[batch_name]['image_size'][0]['height']
            # Write the labels to the respective files
            with open(f'{filename}.txt', 'w') as f:
                for annotation in annotations:
                    class_id = annotation['class_id']
                    x_center = (annotation['left'] + annotation['width'] / 2) / image_width
                    y_center = (annotation['top'] + annotation['height'] / 2) / image_height
                    width = annotation['width'] / image_width
                    height = annotation['height'] / image_height
                    f.write(f'{class_id} {x_center} {y_center} {width} {height}\n')
        
    # Write the labels to the respective files for data collected in batch01
    data01 = load_multiple_json_objects('batch01.json')
    write_labels_to_file(data01)
    
    # Write the labels to the respective files for data collected in batch02
    data02 = load_multiple_json_objects('batch02.json')
    write_labels_to_file(data02)
    
    # Write the labels to the respective files for data collected in batch03
    data03 = load_multiple_json_objects('batch03.json')
    write_labels_to_file(data03)    
                
# Convert images to YOLOv8 format (640x640)
def conversion_to_yolov8_format():
    '''Resize the images to 640x640 for YOLOv8 format.'''

    def convert_image_to_yolov8_format(image_path):
        '''Convert the image to YOLOv8 format.'''
        image = cv.imread(image_path)
        image = cv.resize(image, (640, 640))
        cv.imwrite(image_path, image)

    for d in ['train', 'test', 'valid']:
        for file in tqdm(os.listdir(f'{d}/images'), desc=f'Converting {d} images to YOLOv8 format ...', unit='files'):
            file_path = os.path.join(d, 'images', file)
            convert_image_to_yolov8_format(file_path)

if __name__ == '__main__':
    # organize_image_directories()
    organize_labels_directories()
    # conversion_to_yolov8_format()