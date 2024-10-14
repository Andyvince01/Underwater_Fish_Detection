from .ddpm_trans_modules import diffusion, unet

import torch.nn as nn

def define_G(
        in_channel : int = 6,
        out_channel : int = 3,
        inner_channel : int = 48,
        norm_groups : int = 24,
        channel_mults : tuple = (1, 2, 4, 8, 8),
        attn_res : tuple = (16),
        res_blocks : int = 2,
        dropout : float = 0.2,
        image_size : int = 640
    ) -> diffusion.GaussianDiffusion:
    ''' This function defines the Generator model. 
    
    Parameters
    ----------
    in_channel : int
        Number of input channels.
    out_channel : int
        Number of output channels.
    inner_channel : int
        Number of inner channels.
    norm_groups : int
        Number of normalization groups.
    channel_mults : tuple
        Number of channel multipliers.
    attn_res : tuple
        Number of attention residuals.
    res_blocks : int
        Number of residual blocks.
    dropout : float
        Dropout rate.
    image_size : int
        Image size.
    '''
    # Define the U-Net model
    model = unet.UNet(
        in_channel=in_channel,
        out_channel=out_channel,
        inner_channel=inner_channel,
        norm_groups=norm_groups,
        channel_mults=channel_mults,
        attn_res=attn_res,
        res_blocks=res_blocks,
        dropout=dropout,
        image_size=image_size
    )
    
    # Define the Gaussian diffusion model
    netG = diffusion.GaussianDiffusion(
        denoise_fn=model,
        image_size=image_size,
        channels=3,
        loss_type='l1',    # L1 or L2
        conditional=True,
        schedule_opt={
            'schedule': 'linear',
            "n_timestep": 2000,
            "linear_start": 1e-6,
            "linear_end": 1e-2
        }
    )
    netG = nn.DataParallel(netG)
    return netG
