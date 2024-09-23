''' fishscale_dataset_generator.py
> This script is used to create the fishscale dataset for YOLOv8 training from three different datasets: deepfish, fish4knowledge and ozfish.
'''

from typing import Literal
from tqdm import tqdm
import cv2 as cv
import os, shutil, random

# Set the working directory
os.chdir('Underwater_Fish_Detection/data/') if os.getcwd().split('/')[-1] != 'data' else None

# Create 'images' and 'labels' directories in fishscale_data
# os.makedirs('fishscale_dataset/images', exist_ok=True)
# os.makedirs('fishscale_dataset/labels', exist_ok=True)

os.makedirs('fishscale_dataset/test', exist_ok=True)
os.makedirs('fishscale_dataset/test/images', exist_ok=True)
os.makedirs('fishscale_dataset/test/labels', exist_ok=True)
os.makedirs('fishscale_dataset/train', exist_ok=True)
os.makedirs('fishscale_dataset/train/images', exist_ok=True)
os.makedirs('fishscale_dataset/train/labels', exist_ok=True)
os.makedirs('fishscale_dataset/valid', exist_ok=True)
os.makedirs('fishscale_dataset/valid/images', exist_ok=True)
os.makedirs('fishscale_dataset/valid/labels', exist_ok=True)

# Set the random seed
random.seed(42)

def copy_datasets(flag : bool = True) -> None:
    ''' This function copy all the images and labels from `deepfish`, `fish4knowledge` and `ozfish` datasets to the new `fishscale_dataset`.
    The total number of samples in the new dataset is 10,154 images (and labels). 
    
    Parameters
    ----------
    flag : bool
        If True, the function will copy all the images and labels from deepfish, fish4knowledge and ozfish datasets to the new fishscale_dataset
    '''
    # Return if flag is false
    if not flag : return
    
    # Move all the images and labels to fishscale_data directory
    for d in ['deepfish', 'fish4knowledge', 'ozfish']:
        for dd in ['test', 'train', 'valid']:
            for ddd in ['images', 'labels']:
                desc = f'Copying {d}/{dd}/{ddd} to fishscale_dataset/{ddd} ...'
                for file in tqdm(os.listdir(f'datasets/{d}/{dd}/{ddd}'), desc=desc, leave=False):
                    shutil.copy(f'datasets/{d}/{dd}/{ddd}/{file}', f'fishscale_dataset/{ddd}/{file}')

def dataset_generator(test_set_generator : bool = False) -> None:
    ''' This function generates the train, validation and test datasets from the fishscale_dataset.
    
    Parameters
    ----------
    test_set_generator : bool
        If True, the function will generate a random test set from the fishscale_dataset.
        Otherwise, the function will use the test set used by A. A. Muksit [1] for evaluation and the remaining images for training and validation.
        
    Notes
    -----
    When test_set_generator is False, it is employed the test set used by A. A. Muksit [1] for evaluation (1261 images). So, the remaining images 
    (10,154 - 1261 = 8893) are used for training and validation. The training set is composed of 80% of the remaining images (7114 images) and the
    validation set is composed of 20% of the remaining images (1779 images). In terms of percentage, the fishscale_dataset is divided as follows:
    - Train: 70.1% (7114 images)
    - Validation: 17.5% (1779 images)
    - Test: 12.4% (1261 images)
    
    Instead, when test_set_generator is True, the function generates a random test set from the fishscale_dataset. The dataset is divided as follows:
    - Train: 70%
    - Validation: 15%
    - Test: 15%
    
    References
    ----------
    [1] A. A. Muksit, F. Hasan, M. F. Hasan Bhuiyan Emon, M. R. Haque, A. R. Anwary, and S. Shatabda, “Yolo-fish: A robust fish detection model to
    detect fish in realistic underwater environment,” Ecological Informatics, vol. 72, p. 101847, 2022. https://doi.org/10.1016/j.ecoinf.2022.101847.
    '''
    
    def move_files(image_files : list, labels_files : list, source_images = 'images', source_labels = 'labels', dest : Literal["test", "train", "valid"] = "train") -> None:
        '''Inner function to move the images and labels files to the destination folder.

        Parameters
        ----------
        image_files : list
            The list of image files to move.
        labels_files : list
            The list of label files to move.
        source_images : str, optional
            The source folder for the images, by default 'images'.
        source_labels : str, optional
            The source folder for the labels, by default 'labels'.
        dest : Literal[&quot;test&quot;, &quot;train&quot;, &quot;valid&quot;], optional
            The destination folder, by default &quot;train&quot;.
        '''
        for (image, label) in tqdm(zip(image_files, labels_files), desc=f'Moving images to {dest} ...', unit='files'):
            shutil.move(f'fishscale_dataset/{source_images}/{image}', f'fishscale_dataset/{dest}/images/{image}')
            shutil.move(f'fishscale_dataset/{source_labels}/{label}', f'fishscale_dataset/{dest}/labels/{label}')

    # Generate Test Set (deepfish + ozfish) used by A. A. Muksit
    if test_set_generator:
        test_population = os.listdir('fishscale_dataset/test/')
        test_images = [image for image in os.listdir('fishscale_dataset/images') if image.rsplit('.', 1)[0] + ".txt" in test_population]
        move_files(image_files=test_images, labels_files=test_labels, source_labels='test', dest='test')

    test_images, test_labels = [os.listdir(f'fishscale_dataset/test/{item}') for item in ['images', 'labels']]
    
    k_train, k_val, k_test = (0.7, 0.5, 1) if len(test_images) == 0 else (0.8, 1.0, 0)
    
    # Generate Training Set
    train_population = [image for image in os.listdir('fishscale_dataset/images') if image not in test_images]
    train_images = random.sample(population=train_population, k=int(k_train * len(train_population)))
    train_images_names = set(image.rsplit('.', 1)[0] for image in train_images)
    train_labels = [file for file in os.listdir('fishscale_dataset/labels') if file.rsplit('.', 1)[0] in train_images_names]
    move_files(image_files=train_images, labels_files=train_labels, dest='train')
    
    # # Generate Validation Set
    val_population = [image for image in os.listdir('fishscale_dataset/images') if image not in train_images and image not in test_images]
    val_images = random.sample(population=val_population, k=int(k_val * len(val_population)))
    val_images_names = set(image.rsplit('.', 1)[0] for image in val_images)
    val_labels = [file for file in os.listdir('fishscale_dataset/labels') if file.rsplit('.', 1)[0] in val_images_names]
    move_files(image_files=val_images, labels_files=val_labels, dest='valid')
    
    # Generate Test Set
    test_population = [image for image in os.listdir('fishscale_dataset/images') if image not in train_images and image not in val_images]
    test_images = random.sample(population=test_population, k=int(k_test * len(test_population)))
    test_images_names = set(image.rsplit('.', 1)[0] for image in test_images)
    test_labels = [file for file in os.listdir('fishscale_dataset/labels') if file.rsplit('.', 1)[0] in test_images_names]
    move_files(image_files=test_images, labels_files=test_labels, dest='test')

    # Delete the remaining images and labels (if any)
    for image, label in zip(os.listdir('fishscale_dataset/images'), os.listdir('fishscale_dataset/labels')):
        os.remove(f'fishscale_dataset/images/{image}')
        os.remove(f'fishscale_dataset/labels/{label}')
    os.rmdir('fishscale_dataset/images')
    os.rmdir('fishscale_dataset/labels')
    
if __name__ == '__main__':
    # copy_datasets(flag=False)
    # dataset_generator(test_set_generator=False)
    
    print(len(os.listdir(os.path.join(os.getcwd(), 'fishscale_dataset/train/images/'))))
    print(len(os.listdir(os.path.join(os.getcwd(), 'fishscale_dataset/train/labels/'))))
    
    print(len(os.listdir(os.path.join(os.getcwd(), 'fishscale_dataset/valid/images/'))))
    print(len(os.listdir(os.path.join(os.getcwd(), 'fishscale_dataset/valid/labels/'))))

    print(len(os.listdir(os.path.join(os.getcwd(), 'fishscale_dataset/test/images/'))))
    print(len(os.listdir(os.path.join(os.getcwd(), 'fishscale_dataset/test/labels/'))))

    