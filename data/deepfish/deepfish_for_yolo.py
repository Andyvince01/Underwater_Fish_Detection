''' deepfish_for_yolo.py 
> This script is used to convert the deepfish dataset to yolo format.
'''

from tqdm import tqdm
import cv2 as cv
import os, shutil

# Define the function to organize the dataset directories
def organize_dataset_directories():
    '''Organize the dataset directories for YOLO format.'''
    
    # Set the working directory
    if os.getcwd().split('\\')[-1] != 'deepfish':
        os.chdir('data/deepfish')

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

    # Define train and valid for 'Nagative_samples' directory
    os.makedirs('Nagative_samples/train', exist_ok=True)
    os.makedirs('Nagative_samples/valid', exist_ok=True)

    for f in os.listdir('Nagative_samples'):
        for sf in os.listdir(f'Nagative_samples/{f}'):
            if sf.endswith('.jpg'):
                shutil.move(f'Nagative_samples/{f}/{sf}', f'train/images/{sf}')
            elif sf.endswith('.txt'):
                shutil.move(f'Nagative_samples/{f}/{sf}', f'train/labels/{sf}')
        os.rmdir(f'Nagative_samples/{f}')
    os.rmdir('Nagative_samples')
        
    # Move the images and labels to the respective directories
    for d in os.listdir():
        # Skip the files
        if os.path.isfile(d) or d in dirs:
            continue
        
        # Move train and val images and labels
        for f in ['train', 'valid']:
            for file in os.listdir(os.path.join(d, f)):
                if file.endswith('.jpg'):
                    shutil.move(os.path.join(d, f, file), f'{f}/images/{file}')
                elif file.endswith('.txt'):
                    shutil.move(os.path.join(d, f, file), f'{f}/labels/{file}')     

        # Remove d directory
        os.rmdir(os.path.join(d, 'train'))
        os.rmdir(os.path.join(d, 'valid'))
        os.rmdir(d)

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
    organize_dataset_directories()
    conversion_to_yolov8_format()