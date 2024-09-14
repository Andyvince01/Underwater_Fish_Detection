''' 
    > This file contains utility functions for data processing.
'''


from PIL import Image
from tqdm import tqdm
from typing import Tuple

import os

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640
DPI = 96

os.chdir('data/fishscale_dataset')

def size_distribution(folder : str = "train", save : bool = True) -> None:
    ''' This function computes the size distribution of the fish in the images. 
    
    Parameters
    ----------
    folder : str
        The folder to process. By default, the folder is set to 'train'.
    save : bool
        Whether to save the plot. By default, the plot is saved.
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
    
    img_size = []
    for image in tqdm(os.listdir(f"{folder}/labels"), desc="Processing images"):
        # Check if the file is a text file (label file)
        if not image.endswith(".txt"): continue
    
        # Open the image and for each fish in the image, get the size
        with open(f"{folder}/labels/" + image, "r") as file:
            for line in file:
                line = line.split()
                img_size.append(get_size(float(line[3]), float(line[4])))
    
    # Show distribution with seaborn
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    from pandas import DataFrame
    
    df = DataFrame(img_size, columns=["Width (cm)", "Height (cm)"])
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(9, 9))
    
    # Use scatterplot with custom styling
    sns.scatterplot(
        data=df,
        x="Width (cm)",
        y="Height (cm)",
        marker='s',  # Square markers
        edgecolor='black',  # No border color
        s=30,  # Size of markers
        color='mediumblue',  # Marker color
        alpha=0.5  # Transparency of markers
    )

    # Enhance plot aesthetics
    ax.set_title(f"FishScale {folder}-set - Fish Size Distribution", fontsize=21, weight='bold', pad=20, fontname='Comic Sans MS')
    ax.set_xlabel("Width (cm)", fontsize=14, labelpad=10, fontname='Comic Sans MS')
    ax.set_ylabel("Height (cm)", fontsize=14, labelpad=10, fontname='Comic Sans MS')
    
    # Set the x and y axis limits
    ax.set_xlim(-0.5, 17.5)
    ax.set_ylim(-0.5, 17.5)
    
    # Set a cream background color for the plot
    fig.patch.set_facecolor('white')  # Background color of the plot area
    ax.set_facecolor('ivory')    # Background color of the plotting area
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    
    # Show the plot
    plt.tight_layout()
    # plt.show()
    
    # Save the plot
    if not save: return
    
    # Save the plot as a PNG file
    fig.savefig(f"../utils/{folder}_size_distribution.pdf", dpi=360)
    
if __name__ == "__main__":
    size_distribution(folder='train', save=True)
    size_distribution(folder='test', save=True)
    size_distribution(folder='valid', save=True)