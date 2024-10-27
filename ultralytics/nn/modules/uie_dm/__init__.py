import os, torch, torch.nn as nn
from typing import Generator
from torchvision import transforms

# Import the DDPM model from the model module:
from .model import DDPM
from .utils import colorstr

# Define the path to the weights directory:
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')

class UieDM(nn.Module):
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
        super(UieDM, self).__init__()
        # Initialize the DDPM model
        self.ddpm = DDPM(phase=phase, weights=os.path.join(WEIGHTS_DIR, weights)).half()
        # Set the model to evaluation mode
        self.ddpm.set_new_noise_schedule(
            schedule_opt={
                "schedule": "linear",
                "n_timestep": 2000,
                "linear_start": 1e-6,
                "linear_end": 1e-2
            }, 
            schedule_phase='val'
        )
        self.ddpm.logger.info(f'🖼️ {colorstr("magenta", "bold", " UieDM")} model initialized!')
                
        self.ddpm.set_requires_grad(self.ddpm.netG, requires_grad=False)
        
    @torch.no_grad()
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        ''' Forward pass of the UieDM model
        
        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        
        Returns
        -------
        torch.Tensor
            The output tensor.
        '''
        from torch.cuda.amp import autocast
        # Define the pre-processing and post-processing transformations
        pre_processing = transforms.Lambda(lambda x: (x - 1) * 2)
        post_processing = transforms.Lambda(lambda x: (x + 1) / 2)

        # Apply the pre-processing transformation to the input tensor
        x = pre_processing(x)        
        # Get the device and dtype of the input tensor
        device = x.device; dtype = x.dtype
        x = x.to(dtype=torch.float32) if dtype != torch.float32 else x
        
        # Perform the forward pass of the model
        with autocast():
            # Feed the input tensor to the model
            self.ddpm.feed_data(x)
            # Perform the test operation on the model
            x = self.ddpm.test(continous=False).unsqueeze(0) if x.dim() == 3 else self.ddpm.test(continous=False)
            
        # Apply the post-processing transformation to the output tensor
        x = post_processing(x)
        x = x.to(dtype=dtype) if dtype != torch.float32 else x
    
        # Return the output tensor
        return x.to(device) if device == torch.device('cpu') else x
        
    def parameters(self) -> Generator:
        ''' Return the parameters of the UieDM model 
        
        Returns
        -------
        Generator
            The generator object containing the parameters of the model.
        '''
        for param in self.ddpm.netG.parameters():
            yield param
            
    def quantize_model(self):
        ''' Apply post-training quantization to the netG model '''
        self.ddpm.netG = torch.quantization.quantize_dynamic(
            self.ddpm.netG, {torch.nn.Linear}, dtype=torch.qint8  # Applica la quantizzazione alle layer Linear
        )
        self.ddpm.logger.info(f'🔄 {colorstr("green", "bold", "Quantization")} applied to netG!')


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