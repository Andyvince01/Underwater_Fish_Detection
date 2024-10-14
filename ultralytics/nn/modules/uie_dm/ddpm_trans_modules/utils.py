''' utils.py
>>> Utility functions for Gaussian Diffusion Model
'''

import math, torch
import numpy as np

from inspect import isfunction

# Function to generate beta schedule
def make_beta_schedule(schedule : str = 'linear', n_timestep : int = 1000, linear_start : float = 1e-4, linear_end : float = 2e-2, cosine_s : float = 8e-3) -> np.ndarray:
    ''' Function to generate beta schedule for diffusion model. 
    
    Parameters
    ----------
    schedule : str
        Schedule type for beta values. Default is 'linear'.
    n_timestep : int
        Number of timesteps for the diffusion model. Default is 1000.
    linear_start : float
        Start value for linear schedule. Default is 1e-4.
    linear_end : float
        End value for linear schedule. Default is 2e-2.
    cosine_s : float
        Cosine value for schedule. Default is 8e-3.

    Returns
    -------
    np.ndarray
        Beta schedule for diffusion model.    
    '''
    # Inner function to generate beta schedule
    def warmup_beta(linear_start : float, linear_end : float, n_timestep : int, warmup_frac : float) -> np.ndarray:
        ''' Warmup beta schedule from linear_start to linear_end over warmup_frac of the total timesteps 
        The remaining timesteps are set to linear_end, which is the default value for the rest of the training.
        
        Parameters
        ----------
        linear_start : float
            The initial value of beta
        linear_end : float
            The final value of beta
        n_timestep : int
            The total number of timesteps
        warmup_frac : float
            The fraction of timesteps over which beta is linearly increased from linear_start to linear_end
            
        Returns
        -------
        betas : np.ndarray
            The beta values for each timestep
        '''
        # Initialize beta values with linear_end for all timesteps (size = n_timestep)
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
        # Calculate the warmup time (number of timesteps over which beta is linearly increased)
        warmup_time = int(n_timestep * warmup_frac)
        # Set the beta values for the warmup time. Linearly increase from linear_start to linear_end
        betas[:warmup_time] = np.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
        # Return the beta values
        return betas
       
    # Generate beta schedule based on the schedule type provided                
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = warmup_beta(linear_start, linear_end, n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = warmup_beta(linear_start, linear_end, n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        # Compute the timestep values for cosine schedule
        timesteps = (torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s)
        # Compute the alphas and betas values for cosine schedule
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    # Return the beta values
    return betas

# Gaussian Diffusion Trainer Class
def exists(x : any) -> bool:
    ''' Function to check if the input is not None.
    
    Parameters
    ----------
    x : any
        Input to check
    
    Returns
    -------
    bool
        True if x is not None, False otherwise
    '''
    return x is not None


def default(val : any, d : any) -> any:
    ''' Function to return the default value if the input is None.
    
    Parameters
    ----------
    val : any
        Input value
    d : any
        Default value
    
    Returns
    -------
    any
        val if it is not None, d otherwise
    '''
    # Return val if it is not None, otherwise return d
    if exists(val): return val
    # Return d if it is a function, otherwise return d
    return d() if isfunction(d) else d

def extract(a : torch.Tensor, t : torch.Tensor, x_shape : tuple) -> torch.Tensor:
    ''' Function to extract coefficients from a based on t and reshape to make it broadcastable with x_shape.
    
    Parameters
    ----------
    a : torch.Tensor
        Coefficients tensor
    t : torch.Tensor
        Time tensor
    x_shape : tuple
        Shape of the input tensor
    
    Returns
    -------
    torch.Tensor
        Extracted coefficients tensor
    '''
    # Get the batch size
    bs, = t.shape
    # Check if the shape of the input tensor is correct
    assert x_shape[0] == bs, f"First dimension of x_shape must be the batch size, but got x_shape[0] = {x_shape[0]} and bs = {bs}!"
    # Check if the coefficients tensor is a tensor
    if not torch.is_tensor(a):
        a = torch.tensor(a, dtype=torch.float, device=t.device)

    # Extract the coefficients based on t
    out = torch.gather(a, 0, t.long())
    assert out.shape == (bs,), f"Expected shape (bs,) = {bs}, but got {out.shape}!"
    
    # Reshape the coefficients tensor to make it broadcastable with x_shape (add dimensions of size 1)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out

def noise_like(shape : tuple, device : torch.device, repeat : bool = False) -> torch.Tensor:
    ''' Function to generate noise tensor with the given shape and device.
    
    Parameters
    ----------
    shape : tuple
        Shape of the noise tensor
    device : torch.device
        Device to store the noise tensor
    repeat : bool
        Flag to repeat the noise tensor along the batch dimension
    
    Returns
    -------
    torch.Tensor
        Noise tensor
    '''
    # Inner function to repeat the noise tensor along the batch dimension
    def repeat_noise() -> torch.Tensor:
        ''' Inner function to repeat the noise tensor along the batch dimension. 
        
        Returns
        -------
        torch.Tensor
            Repeated noise tensor along the batch dimension.
        '''
        return torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    # Inner function to generate noise tensor
    def noise() -> torch.Tensor:
        ''' Inner function to generate noise tensor.
        
        Returns
        -------
        torch.Tensor
            Noise tensor.
        '''
        return torch.randn(shape, device=device)

    # Return the noise tensor based on the repeat flag
    return repeat_noise() if repeat else noise()