''' data.utils.utils.py
>>> 💥 This file contains utility functions for data processing.
'''
from pandas import DataFrame
from PIL import Image
from tqdm import tqdm
from typing import Literal, Tuple

import matplotlib.pyplot as plt, os, seaborn as sns, pandas as pd
import argparse

#--- Constants ---#
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640

#---------------------------------------------------------------------------------------------------------------------#
# 1. Function to compute the number of fish in the images for a given input dimension                                 #
#---------------------------------------------------------------------------------------------------------------------#
def fish_count(fish_sizes : dict, width : float = float('inf'), height : float = float('inf')) -> dict:
    ''' This function computes the number of fish in the images for a given input dimension.
    
    Parameters
    ----------
    fish_sizes : dict
        A dictionary containing the fish sizes for the different sources.
    width : float
        The width of the fish in cm. By default, the width is set to infinity.
    height : float
        The height of the fish in cm. By default, the height is set to infinity.
        
    Returns
    -------
    dict
        A dictionary containing the number of fish in the images for a given input dimension.
    '''
    # Initialize the fish count dictionary
    total_fish = {k: 0 for k in fish_sizes.keys()}
    # Iterate over the sources
    for source in fish_sizes.keys():
        total_fish[source] = sum([1 for fish in fish_sizes[source] if fish[0] <= width and fish[1] <= height])
    return total_fish

#---------------------------------------------------------------------------------------------------------------------#
# 2. Function to compute the size distribution of the fish in the images                                              #
#---------------------------------------------------------------------------------------------------------------------#
def size_distribution(save : bool = True) -> dict:
    ''' This function computes the size distribution of the fish in the images. 
    
    Parameters
    ----------
    save : bool
        Whether to save the plot. By default, the plot is saved.
        
    Returns
    -------
    dict
        A dictionary containing the fish sizes for the different sources.
    '''
    def get_size(width : float, height : float, dpi : int = 96) -> Tuple[float, float]:
        ''' This function returns the size in cm of the fish in the image. 
        
        Parameters
        ----------
        width : float
            The width of the fish in pixels (that is, the width of the bounding box).
        height : float
            The normalized height of the fish (that is, the height divided by the height of the image).
        dpi : int
            The resolution of the plot. By default, the resolution is set to 96.
        
        Returns
        -------
        Tuple[float, float]
            A tuple containing the height and width of the fish in cm.
        '''
        #--- Calculate the size of the fish in cm ---#
        width_cm  = width  * 2.54 / dpi * IMAGE_WIDTH
        height_cm = height * 2.54 / dpi * IMAGE_HEIGHT
        return height_cm, width_cm
    
    #--- Initialize the fish sizes dictionary ---#
    fish_sizes = {k: list() for k in ['train', 'test', 'valid']}

    #--- Iterate over the sources ---#
    for folder in ['train', 'test', 'valid']:
        #--- Iterate over the images ---#
        for label in tqdm(os.listdir(f"{folder}/labels"), desc="Processing images", leave=False):
            # Check if the file is a text file (label file)
            if not label.endswith(".txt"): continue

            with Image.open(f"{folder}/images/" + label.replace(".txt", ".jpg")) as img:
                # Get the DPI of the image
                dpi = img.info['dpi'][0] if 'dpi' in img.info else 96
        
            # Open the image and for each fish in the image, get the size
            with open(f"{folder}/labels/" + label, "r") as file:
                for line in file:
                    line = line.split()
                    height, width = get_size(width=float(line[3]), height=float(line[4]), dpi=dpi)
                    fish_sizes[folder].append((width, height)) 
    
        # Check if the fish sizes should be saved on a plot or not
        if not save: continue
    
        #--- 
        df = DataFrame(fish_sizes[folder], columns=["Width (cm)", "Height (cm)"])
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(9, 9))
        
        # Create a histogram plot of the fish sizes
        sns.histplot(
            data=df,                        # Data source           
            x="Width (cm)",                 # X-axis variable
            y="Height (cm)",                # Y-axis variable
            color='deepskyblue',            # Marker color
            bins=100,                       # Number of bins
            pmax=0.5                        # Maximum density value
        )

        # Enhance plot aesthetics
        ax.set_title(f"FishScale {folder}-set - Fish Size Distribution", fontsize=21, weight='bold', pad=20, fontname='Comic Sans MS')
        ax.set_xlabel("Width (cm)", fontsize=15, labelpad=10, fontname='Comic Sans MS')
        ax.set_ylabel("Height (cm)", fontsize=15, labelpad=10, fontname='Comic Sans MS')
        
        # Set the x and y axis limits
        ax.set_xlim(-0.5, 17.5)
        ax.set_ylim(-0.5, 17.5)
        
        # Set a cream background color for the plot
        fig.patch.set_facecolor('white')                                        # Background color of the plot area
        # ax.set_facecolor('ivory')                                               # Background color of the plotting area
        plt.grid(True, which='both', linestyle='--', linewidth=0.7)
                
        # Save the plot as a PNG file
        fig.savefig(os.path.join('..', 'utils', os.getcwd().split('\\')[-1] + f"_{folder}_size_distribution.pdf"), dpi=360)
        
    return fish_sizes
    
if __name__ == "__main__":
    # Define the command line arguments
    parser = argparse.ArgumentParser('Calculate dataset statistics.')
    # Add the arguments to the parser
    parser.add_argument(
        '--dataset',
        type=str,
        default='fishscale_dataset2',
        help='The dataset to calculate the statistics for. By default, the dataset is set to "fishscale_dataset".',
        choices=['datasets/deepfish', 'datasets/fish4knowledge', 'datasets/ozfish', 'fishscale_dataset', 'fishscale_dataset_2']
    )
    # Parse the arguments
    args = parser.parse_args()
    
    # Set the working directory
    os.chdir(os.path.join('data', args.dataset)) if os.getcwd().split('\\')[-1] != 'data' else os.chdir(args.dataset)
    
    # Compute the size distribution of the fish in the images
    fish_sizes = size_distribution(save=True)
    
    # Total number of fish in all the images
    total_number_of_fish_per_source = {source: len(fish_sizes[source]) for source in fish_sizes.keys()}
    total_number_of_fish_in_all_images = sum(total_number_of_fish_per_source.values())
    
    print(f"Total number of fish per source: {total_number_of_fish_per_source}")
    print(f"Total number of fish in all the images: {total_number_of_fish_in_all_images}")
    
    # Compute the number of fish in the images for a given input dimension
    width, height = 1.5, 1.5
    total_fish = fish_count(fish_sizes, width=width, height=height)
    print(f"Total number of fish in the images having a width less than {width} cm and a height less than {height} cm: {sum(total_fish.values())} ({sum(total_fish.values())/total_number_of_fish_in_all_images*100:.2f}%) :\n\
    • Train: {total_fish['train']} ({total_fish['train']/total_number_of_fish_per_source['train']*100:.2f}%)\n\
    • Test: {total_fish['test']} ({total_fish['test']/total_number_of_fish_per_source['test']*100:.2f}%)\n\
    • Valid: {total_fish['valid']} ({total_fish['valid']/total_number_of_fish_per_source['valid']*100:.2f}%)"
    )   