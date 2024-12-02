''' 💥 src.utils.frame_utils.py
>>>  This script is used to define utility functions to handle frames.
'''

#------------------------------------------------------------------------------------------------------------#
# 1. ⚙️ IMPORTS LIBRARIES AND MODULES
#------------------------------------------------------------------------------------------------------------#
import numpy as np

from pathlib import Path
from PIL import Image

#------------------------------------------------------------------------------------------------------------#
# 2. 🚀 FUNCTIONS
#------------------------------------------------------------------------------------------------------------#

#--- Define the function to convert a numpy array to a PIL image ---#
def img(frame) -> Image.Image:
    ''' Convert a numpy array to a PIL image.
    
    Parameters
    ----------
        frame (np.ndarray): The numpy array to convert.
        
    Returns
    -------
        Image.Image: The PIL image.
    '''
    if isinstance(frame, Image.Image):
        return frame
    elif isinstance(frame, Path):
        return Image.open(frame)
    elif isinstance(frame, np.ndarray):
        return Image.fromarray(frame)
    else:
        raise ValueError(f"Unsupported type: {type(frame)}")

#--- Define the function to crop an image to a bounding box ---#
def crop(frame : np.ndarray, box) -> np.ndarray:
    ''' Crop an image and return the cropped image.

    Parameters
    ----------
    frame (np.ndarray): The image to crop.
    box (tuple | list): The bounding box to use for cropping.
        
    Returns
    -------
        np.ndarray 
            The cropped image.
    '''
    #--- Extract the bounding box coordinates ---#
    x1, y1, x2, y2 = box
    
    #--- Crop the image ---#
    return frame[y1:y2, x1:x2]

#--- Define the function to adjust a bounding box to the image dimensions ---#
def adjust_box_to_image_dimensions(
    im : np.ndarray,                # The image whose dimensions we do not want to exceed.
    box : tuple,             # The input bounding box.
    scale_factor : int,     # The factor by which to multiply the box dimensions.
    min_proportion : int    # The minimum proportion of the size of the image that the box will be adjusted to.
) -> tuple:
    """
    Return a box that is ideally centered like the input box, and scaled by `scale_factor`.
    But given that the output box has to be at least a `min_proportion` proportion of the image
    and must fit inside it, the box can actually be scaled up or down or translated.
    If the input box lies within the image and scale_factor is 1 or greater, the output box will
    include the input one.

    Parameters
    ----------
        im: numpy.ndarray
            The image whose dimensions we do not want to exceed.
        box: tuple or list
            The input bounding box
        scale_factor: scalar
            The factor by which to multiply the box dimensions.
        min_proportion:
            The minimum proportion of the size of the image that the box will be adjusted to. If im is 300 pixels high, 
            min_proportion is 1/3 and box is 80 pixels high, the output box will be 100 pixels high.

    Returns
    -------
        list
            The clamped box.
    """
    #--- Inner function to adjust one dimension of the bounding box ---#
    def adjust_one_dimension(z1 : int, z2 : int, max_z : int) -> tuple:
        ''' Adjust one dimension of the bounding box. 
        
        Parameters
        ----------
            z1 : int
                The first coordinate of the bounding box.
            z2 : int
                The second coordinate of the bounding box.
            max_z : int
                The maximum coordinate of the bounding box (e.g., the height or width of the image).
                
        Returns
        -------
            tuple
                The adjusted bounding box.
        '''
        #--- Calculate the center of the bounding box ---#
        z_center = (z1 + z2) // 2

        #--- Calculate the span of the bounding box that guarantees that the current_max_z - current_min_z will be at most max_z - 1 ---#
        z_span = min(max_z - 1, max((z2 - z1) * scale_factor, int(max_z * min_proportion)))

        current_min_z = z_center - z_span // 2
        current_max_z = current_min_z + z_span

        #--- Adjust the bounding box to match current_min_z, current_max_z ---#
        if current_min_z < 0:
            current_min_z = 0                               # Set the minimum coordinate to 0
            current_max_z -= current_min_z                  # Adjust the maximum coordinate accordingly to the minimum coordinate
        elif current_max_z >= max_z:
            current_min_z -= current_max_z - max_z + 1      # Adjust the minimum coordinate accordingly to the maximum coordinate
            current_max_z = max_z - 1                       # Set the maximum coordinate to the maximum height or width of the image
        
        #--- Return the adjusted coordinates ---#
        return int(current_min_z), int(current_max_z)

    #--- Make sure the minimum proportion is positive ---#
    min_proportion = abs(min_proportion)

    #--- Extract the height and width of the image ---#
    max_y, max_x = im.shape[:2]
    x1, y1, x2, y2 = box
        
    #--- Adjust the bounding box to the image dimensions ---#    
    x1, x2 = adjust_one_dimension(x1, x2, max_x)
    y1, y2 = adjust_one_dimension(y1, y2, max_y)
    
    #--- Return the adjusted bounding box ---#
    return x1, y1, x2, y2

#--- Define the function to square up a bounding box to a square ---#
def square_up_box(im : np.ndarray, box) -> tuple:
    ''' Square up a bounding box to a square. 
    
    Parameters
    ----------
        im (np.ndarray): The image to square up the bounding box.
        box (tuple | list): The bounding box to square up.
        
    Returns
    -------
        tuple: The squared up bounding box.
    '''
    #--- Extract original image dimensions and bounding box coordinates ---#
    max_y, max_x = im.shape[:2]
    x1, y1, x2, y2 = box
    
    #--- Calculate the difference between the x and y coordinates ---#    
    dx = x2 - x1
    dy = y2 - y1
    diff = abs(dy - dx)
    
    #--- Adjust the bounding box to a square ---#
    if dx > dy:
        if dx > max_y:
            y1 = 0                          # Set the minimum y coordinate to 0
            y2 = max_y - 1                  # Set the maximum y coordinate to the maximum height - 1
        else:
            y1 -= diff // 2                 # Adjust the minimum y coordinate (extend y1 to bottom)
            y2 += (diff + 1) // 2           # Adjust the maximum y coordinate (extend y2 to top)
    else:
        if dy > max_x:
            x1 = 0                          # Set the minimum x coordinate to 0
            x2 = max_x - 1                  # Set the maximum x coordinate to the maximum width - 1
        else:
            x1 -= diff // 2                 # Adjust the minimum x coordinate (extend x1 to left)
            x2 += (diff + 1) // 2           # Adjust the maximum x coordinate (extend x2 to right)
        
    #--- Return the squared up bounding box ---#
    return x1, y1, x2, y2

#--- Define the function to pad an image up to a square ---#
def pad_up_to_square(im : np.ndarray) -> np.ndarray:
    ''' Pad an image up to a square. 
    
    Parameters
    ----------
        im : np.ndarray
            The image to pad up to a square.
            
    Returns
    -------
        np.ndarray
            The padded image up
    '''
    #--- Extract the image dimensions ---#
    shape = [*im.shape]; max_y, max_x = shape[:2]

    #--- If the image is already a square, return the image ---#
    if max_x == max_y: return im
    
    #--- Calculate the total padding amount ---#
    total_pad_amount = abs(max_y - max_x)
    
    #--- Determine the axis to pad the image along ---#
    axis = 1 if max_x < max_y else 0
    
    #--- Pad the image up to a square ---#
    shape[axis] = total_pad_amount // 2
    pad_before = np.full(shape, 114, im.dtype)

    shape[axis] = (total_pad_amount + 1) // 2
    pad_after = np.full(shape, 114, im.dtype)
    
    #--- Return the padded image up to a square ---#
    return np.concatenate([pad_before, im, pad_after], axis=axis)