'''
> This file contains the implementation of the Gaussian Diffusion model.
'''

#===============================================================================
# IMPORT LIBRARIES
#===============================================================================
# Import the required libraries
import torch
import numpy as np
# Import the required modules
from torch import nn
from functools import partial
from tqdm import tqdm
# Import the required custom modules
from .style_transfer import VGGPerceptualLoss
from .utils import default, extract, make_beta_schedule, noise_like

#===============================================================================
# GAUSSIAN DIFFUSION CLASS
#===============================================================================
class GaussianDiffusion(nn.Module):
    ''' Gaussian Diffusion Model Class '''

    def __init__(
        self,
        denoise_fn : nn.Module,
        image_size : int,
        channels : int = 3,
        loss_type : str = 'l1',
        conditional : bool = True,
        schedule_opt : dict = None        
    ) -> None:
        ''' Initialize the Gaussian Diffusion Model Class. 
        
        Parameters
        ----------
        denoise_fn : nn.Module
            The denoising function for the Gaussian diffusion model. Typically, this is a U-Net model.
        image_size : int
            The size of the image.
        channels : int
            The number of channels in the image. Default is 3.
        loss_type : str
            The type of loss function to use. Default is 'l1'.
        conditional : bool
            Flag to indicate if the model is conditional. Default is True.
        schedule_opt : dict
            The schedule options for the diffusion model. Default is None.
        '''
        # Call the super constructor
        super().__init__()
        # Initialize the class variables
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.conditional = conditional
        self.loss_type = loss_type
        self.eta = 0
        self.sample_proc = 'ddim'

    def set_loss(self, device : torch.device) -> None:
        ''' Set the loss function for the model. '''
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss().to(device)
            self.style_loss = VGGPerceptualLoss().to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss().to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt : dict, device : torch.device) -> None:
        ''' Set the new noise schedule for the diffusion model. 
        In other words, this function sets the beta schedule for the diffusion model.
        
        Parameters
        ----------
        schedule_opt : dict
            The schedule options for the diffusion model.
        device : torch.device
            The device to use for the diffusion model.       
        '''
        # Set the loss function for the model        
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        # Generate the beta schedule based on the schedule type provided
        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],              # The schedule type (e.g., 'quad', 'linear', 'warmup10', 'warmup50', 'const', 'jsd', 'cosine')
            n_timestep=schedule_opt['n_timestep'],          # The number of timesteps for the diffusion model (e.g., 1000)
            linear_start=schedule_opt['linear_start'],      # The starting value for the linear schedule (e.g., 0.01)
            linear_end=schedule_opt['linear_end']           # The ending value for the linear schedule (e.g., 0.01)
        )
        # Calculate the alpha values for the diffusion model
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        # Calculate the sigma values for the diffusion model
        ddim_sigma = (self.eta * ((1 - alphas_cumprod_prev) / (1 - alphas_cumprod) * (1 - alphas_cumprod / alphas_cumprod_prev)) ** 0.5)
        self.ddim_sigma = to_torch(ddim_sigma)
        # Set the time steps for the diffusion model
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        # Register the buffers for the diffusion model
        self.register_buffer('betas', to_torch(betas))                                                  # Beta values for the diffusion model
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))                                # Cumulative product of alpha values
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))                      # Cumulative product of previous alpha values
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))                  # Square root of cumulative product of alpha values
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))   # Square root of 1 - cumulative product of alpha values
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))     # Log of 1 - cumulative product of alpha values
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))       # Square root of reciprocal of cumulative product of alpha values
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1))) # Square root of reciprocal of cumulative product of alpha values - 1

        # Compute the posterior variance and mean coefficients for the diffusion model (q(x_{t-1} | x_t, x_0))
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # Above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))                        # Posterior variance for the diffusion model
        # Below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start : torch.Tensor, t : torch.Tensor) -> tuple:
        ''' Compute the mean and variance for the q distribution.
        
        Parameters
        ----------
        x_start : torch.Tensor
            The starting image tensor.
        t : torch.Tensor
            The time step tensor.
            
        Returns
        -------
        tuple
            The mean and variance for the q distribution.
        '''
        # Compute the mean for the q distribution
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        
        # Compute the variance for the q distribution
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)

        # Compute the log variance for the q distribution
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)

        # Return the mean, variance, and log variance
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t : torch.Tensor, t : torch.Tensor, noise : torch.Tensor) -> torch.Tensor:
        ''' Predict the starting image from the noise tensor.
        
        Parameters
        ----------
        x_t : torch.Tensor
            The image tensor at time t.
        t : torch.Tensor
            The time step tensor.
        noise : torch.Tensor
            The noise tensor.
            
        Returns
        -------
        torch.Tensor
            The predicted starting image tensor.
        '''
        # Compute the predicted starting image from the noise tensor
        start_image = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t \
                    - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        return start_image       

    def q_posterior(self, x_start : torch.Tensor, x_t : torch.Tensor, t : torch.Tensor) -> tuple:
        ''' Compute the posterior mean and variance for the q distribution.
        
        Parameters
        ----------
        x_start : torch.Tensor
            The starting image tensor.
        x_t : torch.Tensor
            The image tensor at time t.
        t : torch.Tensor
            The time step tensor.
        
        Returns
        -------
        tuple
            The posterior mean and variance for the q distribution.
        '''
        # Compute the posterior mean for the q distribution
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start \
                       + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t

        # Compute the posterior variance for the q distribution
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)

        # Compute the clipped log variance for the q distribution
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        # Return the posterior mean, variance, and clipped log variance
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x : torch.Tensor, t : torch.Tensor, clip_denoised : bool, condition_x : torch.Tensor = None) -> tuple:
        ''' Compute the mean and variance for the p distribution.
        
        Parameters
        ----------
        x : torch.Tensor
            The image tensor.
        t : torch.Tensor
            The time step tensor.
        clip_denoised : bool
            Flag to indicate if the denoised image should be clipped.
        condition_x : torch.Tensor
            The conditional image tensor. Default is None.
        
        Returns
        -------
        tuple
            The mean and variance for the p distribution.
        '''
        # If the conditional image tensor is not None, use it to compute the denoised image
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(x_t=x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), t))
        # Otherwise, compute the denoised image
        else:
            x_recon = self.predict_start_from_noise(x_t=x, t=t, noise=self.denoise_fn(x, t))

        # If the denoised image should be clipped, clip it
        if clip_denoised: x_recon.clamp_(-1., 1.)

        # Extract the alpha, alpha_prev, sigma, and sqrt_one_minus_alphas values
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        
        # Return the mean and variance for the p distribution
        return model_mean, posterior_log_variance

    def p_mean_variance_ddim(self, x : torch.Tensor, t : torch.Tensor, clip_denoised : bool, condition_x : torch.Tensor = None, style = None) -> torch.Tensor:
        ''' Compute the mean and variance for the p distribution using the DDIM model.
        
        Parameters
        ----------
        x : torch.Tensor
            The image tensor.
        t : torch.Tensor
            The time step tensor.
        clip_denoised : bool
            Flag to indicate if the denoised image should be clipped.
        condition_x : torch.Tensor
            The conditional image tensor. Default is None.
        style : torch.Tensor
            The style tensor. Default is None.
        
        Returns
        -------
        torch.Tensor
            The mean for the p distribution.
        '''
        # If the conditional image tensor is not None, use it to compute the denoised image        
        if condition_x is not None:
            x_recon = self.denoise_fn(torch.cat([condition_x, x], dim=1), t, style)
        # Otherwise, compute the denoised image
        else:
            x_recon = self.denoise_fn(x, t)
            
        # Get the alpha, alpha_prev, sigma, and sqrt_one_minus_alphas values
        alpha = extract(self.alphas_cumprod, t, x_recon.shape)
        alpha_prev = extract(self.alphas_cumprod_prev, t, x_recon.shape)
        sigma = extract(self.ddim_sigma, t, x_recon.shape)
        sqrt_one_minus_alphas = extract(self.sqrt_one_minus_alphas_cumprod, t, x_recon.shape)
        pred_x0 = (x - sqrt_one_minus_alphas * x_recon) / (alpha ** 0.5)
        
        # Compute the predicted starting image from the noise tensor
        dir_xt = torch.sqrt(1. - alpha_prev - sigma ** 2) * x_recon
        noise = torch.randn(x.shape, device=x.device)
        x_prev = (alpha_prev ** 0.5) * pred_x0 + dir_xt + sigma * noise
        
        # Return the predicted starting image
        return x_prev

    @torch.no_grad()
    def p_sample(self, x : torch.Tensor, t : torch.Tensor, clip_denoised : bool = True, repeat_noise : bool = False, condition_x : torch.Tensor = None, style : torch.Tensor = None) -> torch.Tensor:
        ''' Sample the image from the p distribution.
        
        Parameters
        ----------
        x : torch.Tensor
            The image tensor.
        t : torch.Tensor
            The time step tensor.
        clip_denoised : bool
            Flag to indicate if the denoised image should be clipped. Default is True.
        repeat_noise : bool
            Flag to indicate if the noise should be repeated. Default is False.
        condition_x : torch.Tensor
            The conditional image tensor. Default is None.
        style : torch.Tensor
            The style tensor. Default is None.
        
        Returns
        -------
        torch.Tensor
            The sampled image tensor.
        '''
        # Get the shape of the image tensor
        b, *_, device = *x.shape, x.device
        
        # Get the mean and variance for the p distribution
        model_mean, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, style=style)

        # Sample the noise tensor
        noise = noise_like(x.shape, device, repeat_noise)

        # No noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        # Sample the image from the p distribution
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample2(self, x : torch.Tensor, t : torch.Tensor, clip_denoised : bool = True, repeat_noise : bool = False, condition_x : torch.Tensor = None, style : torch.Tensor = None) -> torch.Tensor:
        ''' Sample the image from the p distribution using the DDIM model.
        
        Parameters
        ----------
        x : torch.Tensor
            The image tensor.
        t : torch.Tensor
            The time step tensor.
        clip_denoised : bool
            Flag to indicate if the denoised image should be clipped. Default is True.
        repeat_noise : bool
            Flag to indicate if the noise should be repeated. Default is False.
        condition_x : torch.Tensor
            The conditional image tensor. Default is None.
        style : torch.Tensor
            The style tensor. Default is None.
        
        Returns
        -------
        torch.Tensor
            The sampled image tensor.
        '''
        # Get the beta values for the diffusion model
        bt = extract(self.betas, t, x.shape)
        # Get the alpha values for the diffusion model
        at = extract((1.0 - self.betas).cumprod(dim=0), t, x.shape)
        # Get the log variance for the diffusion model
        logvar = extract(self.posterior_log_variance_clipped, t, x.shape)

        # Compute the weight and denoised image
        weight = bt / torch.sqrt(1 - at)
        et = self.denoise_fn(torch.cat([condition_x, x], dim=1), t, style)

        # Compute the mean and noise for the diffusion model
        mean = 1 / torch.sqrt(1.0 - bt) * (x - weight * et)
        noise = torch.randn_like(x)

        # Compute the mask for the diffusion model
        mask = 1 - (t == 0).float()
        mask = mask.reshape((x.shape[0],) + (1,) * (len(x.shape) - 1))

        # Sample the image from the p distribution
        xt_next = mean + mask * torch.exp(0.5 * logvar) * noise
        xt_next = xt_next.float()
        return xt_next

    @torch.no_grad()
    def p_sample_ddim(self, x : torch.Tensor, t : torch.Tensor, clip_denoised : bool = True, repeat_noise : bool = False, condition_x : torch.Tensor = None, style : torch.Tensor = None) -> torch.Tensor:
        ''' Sample the image from the p distribution using the DDIM model.
        
        Parameters
        ----------
        x : torch.Tensor
            The image tensor.
        t : torch.Tensor
            The time step tensor.
        clip_denoised : bool
            Flag to indicate if the denoised image should be clipped. Default is True.
        repeat_noise : bool
            Flag to indicate if the noise should be repeated. Default is False.
        condition_x : torch.Tensor
            The conditional image tensor. Default is None.
        style : torch.Tensor
            The style tensor. Default is None.
        
        Returns
        -------
        torch.Tensor
            The sampled image tensor.
        '''
        # Get the image tensor for the previous time step
        x_prev = self.p_mean_variance_ddim(x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, style=style)
        
        # Return the sampled image tensor
        return x_prev

    @torch.no_grad()
    def p_sample_ddim2(self, x : torch.Tensor, t : torch.Tensor, t_next : torch.Tensor, clip_denoised : bool = True, repeat_noise : bool = False, condition_x : torch.Tensor = None, style : torch.Tensor = None) -> torch.Tensor:
        ''' Sample the image from the p distribution using the DDIM model.
        
        Parameters
        ----------
        x : torch.Tensor
            The image tensor.
        t : torch.Tensor
            The time step tensor.
        t_next : torch.Tensor
            The next time step tensor.
        clip_denoised : bool
            Flag to indicate if the denoised image should be clipped. Default is True.
        repeat_noise : bool
            Flag to indicate if the noise should be repeated. Default is False.
        condition_x : torch.Tensor
            The conditional image tensor. Default is None.
        style : torch.Tensor
            The style tensor. Default is None.
        
        Returns
        -------
        torch.Tensor
            The sampled image tensor.
        '''
        # Get the alpha values for the diffusion model
        at = extract((1.0 - self.betas).cumprod(dim=0), t, x.shape)

        # Denoise function with or without condition
        et = self.denoise_fn(torch.cat([condition_x, x], dim=1), t) if condition_x is not None else self.denoise_fn(x, t)

        # Compute the mean and noise for the diffusion model
        x0_t = (x - et * (1 - at).sqrt()) / at.sqrt()
        
        # Compute the alpha values for the diffusion model
        at_next = torch.ones_like(at) if t_next is None else extract((1.0 - self.betas).cumprod(dim=0), t_next, x.shape)

        # If eta is 0, use the inversion process
        if self.eta == 0:
            # Compute the next image tensor
            xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
        # If the alpha value is greater than the next alpha value, raise an error
        elif at > at_next:
            raise ValueError('Inversion process is only possible with eta = 0')
        # Otherwise, use the diffusion process
        else:
            # Compute the c1 and c2 values for the diffusion process
            c1 = self.eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            # Compute the next image tensor
            xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * torch.randn_like(x0_t)
        # Return the sampled image tensor
        return xt_next

    @torch.no_grad()
    def p_sample_loop(self, x : torch.Tensor, continous : bool, cand : list = None) -> torch.Tensor:
        ''' Sample the image from the p distribution using the DDIM model.
        
        Parameters
        ----------
        x : torch.Tensor
            The image tensor.
        continous : bool
            Flag to indicate if the sampling is continuous.
        cand : list
            The list of candidate time steps. Default is None.
        
        Returns
        -------
        torch.Tensor
            The sampled image tensor.
        '''
        # Get the device for the image tensor
        device = self.betas.device
        # Set the random seed for the image tensor
        g_gpu = torch.Generator(device=device).manual_seed(44444)
        # Get the shape of the image tensor
        shape = x.shape
        # Get the batch size for the image tensor
        b = shape[0]
        # Sample the image tensor
        img = torch.randn(shape, device=device, generator=g_gpu)

        # Define time steps based on user input or predefined values
        full_time_steps = np.array(cand) if cand is not None else np.array([1898, 1640, 1539, 1491, 1370, 1136, 972, 858, 680, 340])
        time_steps = full_time_steps[::5]               # Sample every 5 time steps (e.g., 1898, 1370, 680). In this way, the inference is faster.
        # Iterate over time steps to sample images
        for j, i in enumerate(time_steps):
            # Get the time step tensor
            t = torch.full((b,), i, device=device, dtype=torch.long)
            # Get the next time step tensor
            t_next = None if j == len(time_steps) - 1 else torch.full((b,), time_steps[j + 1], device=device, dtype=torch.long)
            # Sample the image tensor
            img = self.p_sample_ddim2(img, t, t_next, condition_x=x, style=None)
        # Return the sampled image tensor
        return img

    @torch.no_grad()
    def sample(self, batch_size : int, continous : bool = False) -> torch.Tensor:
        ''' Sample the image from the p distribution using the DDIM model.
        
        Parameters
        ----------
        batch_size : int
            The batch size for the image tensor.
        continous : bool
            Flag to indicate if the sampling is continuous. Default is False.
        
        Returns
        -------
        torch.Tensor
            The sampled image tensor.
        '''
        # Get the image size and channels
        image_size = self.image_size; channels = self.channels
        # Sample the image tensor
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in : torch.Tensor, continous : bool = False, cand : list = None) -> torch.Tensor:
        ''' Perform super resolution on the image tensor.
        
        Parameters
        ----------
        x_in : torch.Tensor
            The image tensor.
        continous : bool
            Flag to indicate if the sampling is continuous. Default is False.
        cand : list
            The list of candidate time steps. Default is None.
        
        Returns
        -------
        torch.Tensor
            The super resolved image tensor.
        '''
        # Return the super resolved image tensor
        return self.p_sample_loop(x_in, continous, cand=cand)

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long))

        return img

    def q_sample_recover(self, x_noisy, t, predict_noise=None):
        # noise = default(noise, lambda: torch.randn_like(x_start))
        return (x_noisy - extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_noisy.shape) * predict_noise) / extract(self.sqrt_alphas_cumprod, t, x_noisy.shape)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # fix gama
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR']
        condition_x = x_in['SR']
        [b, c, h, w] = x_start.shape
        t = torch.randint(0, self.num_timesteps, (b,),
                          device=x_start.device).long()

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, t)
        else:
            x_recon = self.denoise_fn(
                torch.cat([condition_x, x_noisy], dim=1), t)


        loss = self.loss_func(noise, x_recon)

        return loss

    def forward(self, x : torch.Tensor, *args, **kwargs) -> torch.Tensor:
        ''' Forward pass for the Gaussian Diffusion model.
        
        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        
        Returns
        -------
        torch.Tensor
            The output tensor.
        '''
        return self.p_losses(x, *args, **kwargs)