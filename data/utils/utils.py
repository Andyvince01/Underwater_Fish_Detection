''' data.utils.utils.py
>>> 💥 This file contains utility functions for data processing.
'''
from pandas import DataFrame
from PIL import Image
from tqdm import tqdm
from typing import Literal, Tuple

import matplotlib.pyplot as plt, os, seaborn as sns, pandas as pd
import argparse

# Constants for the images
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640
DPI = 96

# Function to count the number of fish in the images for a given input dimension
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

# Function to compute the size distribution of the fish in the images
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
    def get_size(width : float, height : float) -> Tuple[float, float]:
        ''' This function returns the size in cm of the fish in the image. 
        
        Parameters
        ----------
        width : float
            The width of the fish in pixels (that is, the width of the bounding box).
        height : float
            The normalized height of the fish (that is, the height divided by the height of the image).
            
        Returns
        -------
        Tuple[float, float]
            A tuple containing the height and width of the fish in cm.
        '''
        # Convert the height and width from pixels to cm
        width_cm  = width  * 2.54 / DPI * IMAGE_WIDTH
        height_cm = height * 2.54 / DPI * IMAGE_HEIGHT
        return height_cm, width_cm
    
    # List to store the fish sizes
    fish_sizes = {k: list() for k in ['train', 'test', 'valid']}

    # Iterate over the folders
    for folder in ['train', 'test', 'valid']:
        # Iterate over the images in the folder
        for image in tqdm(os.listdir(f"{folder}/labels"), desc="Processing images", leave=False):
            # Check if the file is a text file (label file)
            if not image.endswith(".txt"): continue
        
            # Open the image and for each fish in the image, get the size
            with open(f"{folder}/labels/" + image, "r") as file:
                for line in file:
                    line = line.split()
                    height, width = get_size(float(line[3]), float(line[4]))
                    fish_sizes[folder].append((width, height)) 
    
        # Check if the fish sizes should be saved or not
        if not save: continue
    
        # Create a DataFrame from the fish sizes
        df = DataFrame(fish_sizes[folder], columns=["Width (cm)", "Height (cm)"])
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(9, 9))
        
        # Use scatterplot with custom styling
        sns.scatterplot(
            data=df,                        # Data source           
            x="Width (cm)",                 # X-axis variable
            y="Height (cm)",                # Y-axis variable
            marker='s',                     # Square markers
            edgecolor='black',              # No border color
            s=30,                           # Size of markers
            color='mediumblue',             # Marker color
            alpha=0.5                       # Transparency of markers
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
        ax.set_facecolor('ivory')                                               # Background color of the plotting area
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
        default='fishscale_dataset',
        help='The dataset to calculate the statistics for. By default, the dataset is set to "fishscale_dataset".',
        choices=['datasets/deepfish', 'datasets/fish4knowledge', 'datasets/ozfish', 'fishscale_dataset']
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