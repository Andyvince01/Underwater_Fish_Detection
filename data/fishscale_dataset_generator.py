''' fishscale_dataset_generator.py
> This script is used to create the fishscale dataset for YOLOv8 training from three datasets: deepfish, fish4knowledge and ozfish.
'''

from typing import Literal
from tqdm import tqdm
import cv2 as cv
import os, shutil, random

# Set the working directory
os.chdir('data/') if os.getcwd().split('/')[-1] != 'data' else None

def copy_datasets(flag : bool = True):
    ''' This function copy all the images and labels from deepfish, fish4knowledge and ozfish datasets to the new fishscale_dataset'''
    # Create 'images' and 'labels' directories in fishscale_data
    os.makedirs('fishscale_dataset/images', exist_ok=True)
    os.makedirs('fishscale_dataset/labels', exist_ok=True)

    os.makedirs('fishscale_dataset/test', exist_ok=True)
    os.makedirs('fishscale_dataset/test/images', exist_ok=True)
    os.makedirs('fishscale_dataset/test/labels', exist_ok=True)
    os.makedirs('fishscale_dataset/train', exist_ok=True)
    os.makedirs('fishscale_dataset/train/images', exist_ok=True)
    os.makedirs('fishscale_dataset/train/labels', exist_ok=True)
    os.makedirs('fishscale_dataset/valid', exist_ok=True)
    os.makedirs('fishscale_dataset/valid/images', exist_ok=True)
    os.makedirs('fishscale_dataset/valid/labels', exist_ok=True)

    # Return if flag is false
    if not flag : return
    
    # Move all the images and labels to fishscale_data directory
    for d in ['deepfish', 'fish4knowledge', 'ozfish']:
        for f in ['test', 'train', 'valid']:
            for ff in ['images', 'labels']:
                for file in tqdm(os.listdir(f'datasets/{d}/{f}/{ff}'), desc='Copying {f} {ff} from {d} to fishscale/{f}...', leave=False):
                    shutil.copy(f'datasets/{d}/{f}/{ff}/{file}', f'fishscale_dataset/{ff}/{file}')

def dataset_generator(test_set_generator : bool = True):
    ''' fishscale dataset generator '''
    def move_files(image_files : list, labels_files : list, source_images = 'images', source_labels = 'labels', dest : Literal["test", "train", "valid"] = "train"):
        for (image, label) in tqdm(zip(image_files, labels_files), desc=f'Moving  images to {dest} ...', unit='files'):
            shutil.move(f'fishscale_dataset/{source_images}/{image}', f'fishscale_dataset/{dest}/images/{image}')
            shutil.move(f'fishscale_dataset/{source_labels}/{label}', f'fishscale_dataset/{dest}/labels/{label}')

    # Sample the dataset into 'train', 'validation', and 'test'
    random.seed(42)

    # Generate Test Set (deepfish + ozfish) used by A. A. Muksit
    if test_set_generator:
        test_population = os.listdir('fishscale_dataset/test/')
        test_images = [image for image in os.listdir('fishscale_dataset/images') if image.rsplit('.', 1)[0] + ".txt" in test_population]
        move_files(image_files=test_images, labels_files=test_labels, source_labels='test', dest='test')

    test_images = [image for image in os.listdir('fishscale_dataset/test/images')]
    test_labels = [label for label in os.listdir('fishscale_dataset/test/labels')]
    
    # Generate Training Set
    train_population = [image for image in os.listdir('fishscale_dataset/images') if image not in test_images]
    train_images = random.sample(population=train_population, k=int(0.8 * len(train_population)))
    train_images_names = set(image.rsplit('.', 1)[0] for image in train_images)
    train_labels = [file for file in os.listdir('fishscale_dataset/labels') if file.rsplit('.', 1)[0] in train_images_names]
    move_files(image_files=train_images, labels_files=train_labels, dest='train')

    # # Generate Validation Set
    val_population = [image for image in os.listdir('fishscale_dataset/images') if image not in train_images and image not in test_images]
    val_images = random.sample(population=val_population, k=int(1 * len(val_population)))
    val_images_names = set(image.rsplit('.', 1)[0] for image in val_images)
    val_labels = [file for file in os.listdir('fishscale_dataset/labels') if file.rsplit('.', 1)[0] in val_images_names]
    move_files(image_files=val_images, labels_files=val_labels, dest='valid')
    
    # # Generate Test Set
    # test_population = [image for image in os.listdir('fishscale_dataset/images') if image not in train_images and image not in val_images]
    # test_images = random.sample(population=test_population, k=int(1 * len(test_population)))
    # test_images_names = set(image.rsplit('.', 1)[0] for image in test_images)
    # test_labels = [file for file in os.listdir('fishscale_dataset/labels') if file.rsplit('.', 1)[0] in test_images_names]
    # move_files(image_files=test_images, labels_files=test_labels, dest='test')

    # Delete empty folders
    os.rmdir('fishscale_dataset/images')
    os.rmdir('fishscale_dataset/labels')
    
def convert_png_to_jpg():
    dirs = ['train', 'valid', 'test']
    for d in dirs:
        for file in tqdm(os.listdir(f'fishscale_dataset/{d}/images'), desc=f'Converting {d} images to jpg...', unit='files'):
            if file.endswith('.png'):
                img = cv.imread(f'fishscale_dataset/{d}/images/{file}')
                cv.imwrite(f'fishscale_dataset/{d}/images/{file.rsplit(".", 1)[0]}.jpg', img)
                os.remove(f'fishscale_dataset/{d}/images/{file}')

if __name__ == '__main__':
    copy_datasets(flag=False)
    dataset_generator(test_set_generator=False)
    convert_png_to_jpg()