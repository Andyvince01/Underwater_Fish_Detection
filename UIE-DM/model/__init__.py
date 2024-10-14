import os
from .model import DDPM

# Define the path to the weights directory:
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')

# Define the create_model function to create the model for the Gaussian diffusion model:
def create_model(phase='train', weights = 'I950000_E3369') -> DDPM:
    ''' Create the model for the Gaussian diffusion model 
    
    Parameters
    ----------
    phase : str
        The phase of the model. Default is 'train'.
    weights : str
        The path to the weights directory. Default is I950000_E3369.
    logger : Logger
        The logger object. Default is None.
    
    Returns
    -------
    DDPM
        The model object.
    '''
    model = DDPM(phase=phase, weights=os.path.join(WEIGHTS_DIR, weights))
    return model

# Define the __all__ variable to specify the list of module names to import when using the wildcard import statement:
__all__ = ['DDPM', 'create_model']