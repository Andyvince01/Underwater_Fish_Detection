import os, torch, torch.nn as nn
from torchvision import transforms

# Import the DDPM model from the model module:
from .model import DDPM

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
        self.ddpm = DDPM(phase=phase, weights=os.path.join(WEIGHTS_DIR, weights))
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
        self.ddpm.logger.info('Initial Model Finished')
        
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
        # Define the set of transformations to apply to the input and the output tensors
        pre_processing = transforms.Lambda(lambda x: x * 2 - 1)     # Change the range from [0,1] to [-1,1]
        post_processing = transforms.Lambda(lambda x: (x + 1) / 2)  # Change the range from [-1,1] to [0,1]
            
        # Apply the transformations to the input tensor
        x = pre_processing(x)
        
        # Loop over batch size, and apply transformations to each image individually
        batch_size = x.shape[0]
        outputs = []
        
        for i in range(batch_size):
            # Process one image at a time
            img = x[i:i+1]
            
            # Feed the data to the model
            self.ddpm.feed_data(img)

            # Perform the forward pass
            img = self.ddpm.test(continous=False)
                
            # Store the output
            outputs.append(img.unsqueeze(0))
            
        # Stack the outputs into a batch
        x = torch.cat(outputs, dim=0)
                
        # Apply post-processing on the entire batch
        x = post_processing(x)

        # Return the output tensor
        return x
    
    def parameters(self):
        return self.ddpm.netG.parameters()

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