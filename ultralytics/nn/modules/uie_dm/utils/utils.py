'''
> This file contains the utility functions that are used in the project.
'''

import math, torch
import cv2 as cv
import numpy as np

from torchvision.utils import make_grid
from typing import Tuple

def tensor2img(tensor : torch.Tensor, out_type=np.uint8, min_max : Tuple = (-1, 1)) -> np.ndarray:
    '''
    This function converts a torch Tensor into an image Numpy array.
    
    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor.
    out_type : numpy.dtype, optional
        The output type. The default is np.uint8.
    min_max : tuple, optional
        The minimum and maximum values. The default is (-1, 1).
    
    Returns
    -------
    np.ndarray
        The image as a numpy array.
    
    Notes
    -----
    The input tensor can be 4D(B, (3/1), H, W), 3D(C, H, W), or 2D(H, W), any range, RGB channel order.
    Instead, the output is 3D(H, W, C) or 2D(H, W), [0, 255], np.uint8 (default).
    '''
    # Clamp the tensor values (that is, set the values to the min or max if they are outside the range)
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
    # Normalize the tensor values to the range [0, 1]
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  
    # Get the number of dimensions of the tensor
    n_dim = tensor.dim()

    # If the tensor is 4D    
    if n_dim == 4:
        # Get the number of images
        n_img = len(tensor)
        # Create a grid of images
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        # Transpose the image
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGBù
    # If the tensor is 3D
    elif n_dim == 3:
        # Get the image as a numpy array
        img_np = tensor.numpy()
        # Transpose the image.
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    # If the tensor is 2D
    elif n_dim == 2:
        # Get the image as a numpy array
        img_np = tensor.numpy()
    # If the tensor has a different number of dimensions
    else:
        # Raise a TypeError.
        raise TypeError('Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))        
    
    # If the output type is np.uint8
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()         # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.

    # Return the image as a numpy array
    return img_np.astype(out_type)

def save_image(image : np.ndarray, image_path : str) -> None:
    ''' This function saves an input image to a given image path 
    
    Parameters
    ----------
    image : np.ndarray
        The input image.
    image_path : str
        The path where to save the image.
    '''
    assert image_path, 'The image path is not valid'
    # Save the image to the given path.
    cv.imwrite(image_path, cv.cvtColor(image, cv.COLOR_RGB2BGR))
    
def psnr(img1 : np.ndarray, img2 : np.ndarray) -> float:
    ''' This function calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Parameters
    ----------
    img1 : np.ndarray
        The first image.
    img2 : np.ndarray
        The second image.
    
    Returns
    -------
    float
        The PSNR value.
    '''
    # Convert the images to float64
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    # Calculate the Mean Squared Error (MSE)
    mse = np.mean((img1 - img2)**2)
    # Calculate the PSNR value.
    return 20 * math.log10(255.0 / math.sqrt(mse)) if mse else float('inf')

def ssim(img1 : np.ndarray, img2 : np.ndarray) -> float:
    ''' This function calculates the Structural Similarity Index (SSIM) between two images. 
    
    Parameters
    ----------
    img1 : np.ndarray
        The first image.
    img2 : np.ndarray
        The second image.
    
    Returns
    ------- 
    float
        The SSIM value.
    '''
    # Check if the images have the same dimensions
    assert img1.shape == img2.shape, 'Input images must have the same dimensions.'
    # Check if the images have 2 dimensions
    assert img1.ndim == 2, 'Wrong input image dimensions.'    
    
    # Define the constants
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    
    # Convert the images to float64
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Get the Gaussian kernel
    kernel = cv.getGaussianKernel(11, 1.5)
    
    # Create a window
    window = np.outer(kernel, kernel.transpose())
    
    # Calculate the mean of the first image
    mu1 = cv.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    # Calculate the mean of the second image
    mu2 = cv.filter2D(img2, -1, window)[5:-5, 5:-5]
    
    # Calculate the square of the mean of the first image
    mu1_sq = mu1**2
    # Calculate the square of the mean of the second image
    mu2_sq = mu2**2    
    # Calculate the product of the means of the images
    mu1_mu2 = mu1 * mu2
    
    # Calculate the variance of the first image
    sigma1_sq = cv.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    # Calculate the variance of the second image
    sigma2_sq = cv.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    # Calculate the covariance of the images
    sigma12 = cv.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    # Calculate the SSIM map
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # Return the mean of the SSIM map
    return ssim_map.mean()