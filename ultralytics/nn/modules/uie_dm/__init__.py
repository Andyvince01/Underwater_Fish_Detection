import os, torch
from torchvision import transforms

# Import the DDPM model from the model module:
from .model import DDPM

# Define the path to the weights directory:
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')

class UieDM(DDPM):
    ''' The UieDM model implementation as a subclass of the DDPM model '''
    
    def __init__(self, phase='train', weights = 'I950000_E3369'):
        ''' Initialize the UieDM model 
        
        Parameters
        ----------
        phase : str
            The phase of the model. Default is 'train'.
        weights : str
            The path to the weights directory. Default is I950000_E3369.
        '''
        super(UieDM, self).__init__(phase=phase, weights=os.path.join(WEIGHTS_DIR, weights))
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        ''' Forward pass of the UieDM model 
        
        Parameters
        ----------
        x : Tensor
            The input tensor.
        
        Returns
        -------
        Tensor
            The output tensor.
        '''
        # Define the set of transformations to apply to the input tensor
        transform = transforms.Compose([
            transforms.ToTensor(),   # Convert to tensor and normalize [0,1]
            transforms.Lambda(lambda x: x * 2 - 1)  # Change the range from [0,1] to [-1,1]
        ])
        # Apply the transformations to the input tensor
        x = transform(x)
        
        # Feed the data to the model
        self.feed_data(x.unsqueeze(0))
        
        # Perform the forward pass
        x = self.test(continous=False)
        
        # Return the output tensor
        return x

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
__all__ = ['DDPM', 'create_model', 'UieDM']