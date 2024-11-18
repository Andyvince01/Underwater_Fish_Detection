''' data.utils.utils.py
>>> 💥 This file contains utility functions for data processing.
'''
from pandas import DataFrame
from pathlib import Path
from tqdm import tqdm
from typing import Tuple

import matplotlib.pyplot as plt, numpy as np, os, seaborn as sns
import argparse

#---------------------------------------------------------------------------------------------------------------------#
# 1. 🎈 CONSTANTS
#---------------------------------------------------------------------------------------------------------------------#
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640

#---------------------------------------------------------------------------------------------------------------------#
# 2. 🚀 FUNCTIONS
#---------------------------------------------------------------------------------------------------------------------#

#--- 2.1 - Function to compute the number of fish in the images for a given input dimension ---#
def fish_count(fish_sizes : dict, l_width : float = 32.0, l_height : float = 32.0, u_width : float = 96.0, u_height = 96.0) -> dict:
    ''' This function calculates the number of fish in the images having an area less than
    the given input dimension (width x height).
    
    An area of 32 x 32 pixels is considered as the default input dimension. It corresponds to a 
    `small` fish as defined in the COCO dataset. Instead, an area between 32 x 32 and 96 x 96 pixels
    is considered as a `medium` fish, and an area greater than 96 x 96 pixels is considered as a
    `large` fish.
    
    Parameters
    ----------
    fish_sizes : dict
        A dictionary containing the fish sizes for the different sources.
    l_width : float
        The lower width of the fish in pixels. By default, the lower width is set to 32.0.
    l_height : float
        The lower height of the fish in pixels. By default, the lower height is set to 32.0.
    u_width : float
        The upper width of the fish in pixels. By default, the upper width is set to 96.0.
    u_height : float
        The upper height of the fish in pixels. By default, the upper height is set to 96.0.
        
    Returns
    -------
    dict
        A dictionary containing the number of fish in the images having an area less than the given
        input dimension.
    '''
    return {
        k: sum([1 for fish in fish_sizes[k] if l_width * l_height <= fish[0] * fish[1] < u_width * u_height])
        for k in fish_sizes.keys()
    }
    
#--- 2.2 - Function to compute the size distribution of the fish in the images ---#
def size_distribution(dataset='fishscale_dataset', save : bool = True) -> dict:
    ''' This function computes the size distribution of the fish in the images. 
    
    Parameters
    ----------
    dataset : str
        The dataset to calculate the statistics for. By default, the dataset is set to "fishscale_dataset".
    save : bool
        Whether to save the plot. By default, the plot is saved.
        
    Returns
    -------
    dict
        A dictionary containing the fish sizes for the different sources.
    '''
    #--- Inner function to get the size of the fish in cm ---#
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
    empty_fish = {k : 0.0 for k in ['train', 'test', 'valid']}
    for folder in ['train', 'test', 'valid']:
        #--- Iterate over the images ---#
        for label in tqdm(os.listdir(f"{ROOT/dataset/folder}/labels"), desc="Processing images", leave=False):
            # Check if the file is a text file (label file)
            if not label.endswith(".txt"): continue

            # Open the image and for each fish in the image, get the size
            with open(f"{ROOT/dataset/folder}/labels/" + label, "r") as file:
                # Check if the file is empty (no fish in the image)                            
                if os.stat(f"{ROOT/dataset/folder}/labels/" + label).st_size == 0: empty_fish[folder] += 1; continue
                # Iterate over the lines in the file and get the fish sizes                
                for line in file:
                    line = line.split()
                    height, width = float(line[3]) * IMAGE_HEIGHT, float(line[4]) * IMAGE_WIDTH
                    fish_sizes[folder].append((width, height)) 
    
        # Check if the fish sizes should be saved on a plot or not
        if not save: continue
        
        #--- Plot the size distribution of the fish ---#
        plot_data(fish_sizes[folder], dataset + "_" + folder)
                
    return fish_sizes

#--- 2.3 - Function to plot the size distribution of the fish ---#
def plot_data(data : list, file_name : str):
    ''' This function plots the size distribution of the fish in the images. 
    
    Parameters
    ----------
    data : list
        A list containing the fish sizes.
    folder : str
        The folder name.
    '''
    #--- Create a DataFrame from the data ---#
    df = DataFrame(data, columns=['Width (px)', 'Height (px)'])
    
    #--- Plot the size distribution of the fish ---#
    plt.figure(figsize=(12, 12))
    
    #--- Plot the size distribution of the fish ---#
    sns.histplot(
        data=df,                        # Data source           
        x="Width (px)",                 # X-axis variable
        y="Height (px)",                # Y-axis variable
        color='deepskyblue',            # Marker color
        bins=100,                       # Number of bins
        pmax=0.5,                       # Maximum density value
        zorder=1                        # Plot order
    )
    
    #--- Curve for the small, medium, and large fish ---#
    x = np.linspace(1, IMAGE_WIDTH, 1000)
    y_small = 1024/x  # area = 32*32 = 1024
    y_medium = 9216/x  # area = 96*96 = 9216
    
    # Plotting the curves
    plt.plot(x, y_small, 'r--', alpha=0.5, zorder=2)
    plt.plot(x, y_medium, 'b--', alpha=0.5, zorder=2)

    # Fill the areas under the curves
    plt.fill_between(x, 0, y_small, alpha=0.1, color='red', label='Small (area < $32^2$)', zorder=-1)
    plt.fill_between(x, y_small, y_medium, alpha=0.1, color='blue', label='Medium ($32^2 \leq$ area < $96^2$)', zorder=-1)
    plt.fill_between(x, y_medium, IMAGE_WIDTH, alpha=0.1, color='green', label='Large (area $\geq 32^2$)', zorder=-1)

    #--- Set the plot properties ---#

    # Set the axis limits and labels
    plt.xlim(0, IMAGE_WIDTH)
    plt.ylim(0, IMAGE_HEIGHT)
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    
    # Set the aspect ratio of the plot
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    # Legend    
    plt.legend(loc='best', fontsize=15)

    #--- Reference points ---#
    reference_points = [(32, 32, 'small'), (96, 96, 'medium')]
    for x, y, size in reference_points:
        if size == 'small': color = 'red'
        elif size == 'medium': color = 'blue'
        else: color = 'green'
        plt.plot(x, y, 'o', color=color, markersize=3, alpha=0.3)
    
    #--- Save the plot ---#
    plt.savefig(f"{ROOT/'utils'/'images'/ file_name}_size_distribution.pdf", dpi=360)

#---------------------------------------------------------------------------------------------------------------------#
# 3. 🎯 MAIN
#---------------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    # Define the command line arguments
    parser = argparse.ArgumentParser('Calculate dataset statistics.')
    # Add the arguments to the parser
    parser.add_argument(
        '--dataset',
        type=str,
        default='fishscale_dataset',
        help='The dataset to calculate the statistics for. By default, the dataset is set to "fishscale_dataset".',
        choices=['datasets/deepfish', 'datasets/fish4knowledge', 'datasets/ozfish', 'fishscale_dataset', 'fishscale_dataset_2']
    )
    # Parse the arguments
    args = parser.parse_args()
        
    # Compute the size distribution of the fish in the images
    fish_sizes = size_distribution(dataset=args.dataset, save=True)
    
    # Total number of fish in all the images
    total_number_of_fish_per_source = {source: len(fish_sizes[source]) for source in fish_sizes.keys()}
    total_number_of_fish_in_all_images = sum(total_number_of_fish_per_source.values())
    
    print(f"- Total number of fish per source: {total_number_of_fish_per_source}")
    print(f"- Total number of fish in all the images: {total_number_of_fish_in_all_images}")
    
    #--- Compute the number of fish classified as `SMALL` according to the COCO dataset ---#
    l_width, l_height = 0, 0; u_width, u_height = 10, 10
    total_fish = fish_count(fish_sizes, l_width=l_width, l_height=l_height, u_width=u_width, u_height=u_height)
    print(f"- Total number of fish in the images having an area between {l_width}x{l_height} and {u_width}x{u_height}: {sum(total_fish.values())} ({sum(total_fish.values())/total_number_of_fish_in_all_images*100:.2f}%) :\n\
        • Train: {total_fish['train']} ({total_fish['train']/total_number_of_fish_per_source['train']*100:.2f}%)\n\
        • Test: {total_fish['test']} ({total_fish['test']/total_number_of_fish_per_source['test']*100:.2f}%)\n\
        • Valid: {total_fish['valid']} ({total_fish['valid']/total_number_of_fish_per_source['valid']*100:.2f}%)"
    ) 

    #--- Compute the number of fish classified as `MEDIUM` according to the COCO dataset ---#    
    l_width, l_height = 0, 0; u_width, u_height = 16, 16
    total_fish = fish_count(fish_sizes, l_width=l_width, l_height=l_height, u_width=u_width, u_height=u_height)
    print(f"- Total number of fish in the images having an area between {l_width}x{l_height} and {u_width}x{u_height}: {sum(total_fish.values())} ({sum(total_fish.values())/total_number_of_fish_in_all_images*100:.2f}%) :\n\
        • Train: {total_fish['train']} ({total_fish['train']/total_number_of_fish_per_source['train']*100:.2f}%)\n\
        • Test: {total_fish['test']} ({total_fish['test']/total_number_of_fish_per_source['test']*100:.2f}%)\n\
        • Valid: {total_fish['valid']} ({total_fish['valid']/total_number_of_fish_per_source['valid']*100:.2f}%)"
    )
    
    #--- Compute the number of fish classified as `MEDIUM` according to the COCO dataset ---#    
    l_width, l_height = 96, 96; u_width, u_height = float('inf'), float('inf')
    total_fish = fish_count(fish_sizes, l_width=l_width, l_height=l_height, u_width=u_width, u_height=u_height)
    print(f"- Total number of fish in the images having an area between {l_width}x{l_height} and {u_width}x{u_height}: {sum(total_fish.values())} ({sum(total_fish.values())/total_number_of_fish_in_all_images*100:.2f}%) :\n\
        • Train: {total_fish['train']} ({total_fish['train']/total_number_of_fish_per_source['train']*100:.2f}%)\n\
        • Test: {total_fish['test']} ({total_fish['test']/total_number_of_fish_per_source['test']*100:.2f}%)\n\
        • Valid: {total_fish['valid']} ({total_fish['valid']/total_number_of_fish_per_source['valid']*100:.2f}%)"
    )