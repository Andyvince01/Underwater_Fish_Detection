''' deepfish_for_yolo.py 
> This script is used to convert the deepfish dataset to yolo format.
'''
import os
import shutil
import cv2 as cv

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

    # def convert_labels_to_yolov8_format(label_path):
    #     '''Convert the labels to YOLOv8 format.'''
    #     with open(label_path, 'r') as f:
    #         lines = f.readlines()
            
    #     with open(label_path, 'w') as f:
    #         for line in lines:
    #             line = line.split()
    #             x_center = float(line[1]) * 640
    #             y_center = float(line[2]) * 640
    #             width = float(line[3]) * 640
    #             height = float(line[4]) * 640
    #             f.write(f'{line[0]} {x_center} {y_center} {width} {height}\n')

    for d in ['train', 'test', 'valid']:
        for f in ['images', 'labels']:
            for file in os.listdir(f'{d}/{f}'):
                file_path = os.path.join(d, f, file)
                if file.endswith('.jpg'):
                    convert_image_to_yolov8_format(file_path)
                # elif file.endswith('.txt'):
                #     convert_labels_to_yolov8_format(file_path)

if __name__ == '__main__':
    organize_dataset_directories()
    conversion_to_yolov8_format()