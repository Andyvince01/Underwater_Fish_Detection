import os, torch, torch.nn as nn, torch.nn.functional as F

from collections import OrderedDict
from .utils import Logger
from .generator import define_G
import logging

class DDPM:
    ''' This class defines the model for the Gaussian diffusion model '''
    def __init__(self, phase='val', weights='weights/I950000_E3369'):
        # Initialize the class variables
        self.phase = phase
        self.weights = weights

        # Set the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize the logger
        self.logger = Logger(logger_name='DDPM', level=logging.INFO, screen=True)
        self.logger.info('Initializing the DDPM model...')
        
        # Define network and load pretrained models
        self.netG = self.set_device(define_G())
        self.schedule_phase = None

        # Set loss and load resume state
        self.set_loss()
        self.loss_func = nn.MSELoss(reduction='sum').to(self.device)
        # Set the new noise schedule for the model
        self.set_new_noise_schedule(
            schedule_opt={
                "schedule": "linear",
                "n_timestep": 2000,
                "linear_start": 1e-6,
                "linear_end": 1e-2
            },
            schedule_phase='val'
        )
        # if self.phase == 'train':
        #     # Train mode
        #     self.netG.train()
        #     # Find the parameters to optimize
        #     optim_params = list(self.netG.parameters())
        #     self.optG = torch.optim.Adam(
        #         params=optim_params,
        #         lr=1e-4
        #     )

        # Define a dictionary to store the logs
        self.log_dict = OrderedDict()
        
        # Load the pretrained models
        self.load_network()
        # self.print_network()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self, flag=None):
        # need to average in multi-gpu
        if flag is None:

            self.optG.zero_grad()
            l_pix = self.netG(self.data, flag=None)

            l_pix.backward()
            self.optG.step()
            # print('single mse:', l_pix.item())
            # set log
            self.log_dict['l_pix'] = l_pix.item()

    def optimize_parameters2(self):
        # need to average in multi-gpu
        self.optG.zero_grad()
        l_pix = self.netG(self.data)
        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        l_pix = l_pix.sum() / int(b * c * h * w)
        l_pix.backward()
        self.optG.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()


    def test(self, cand=None, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data, continous)
            else:
                # n = None
                # self.temp, miu, var = self.netG_air.super_resolution(
                #     self.data, continous, flag='style', n=n)
                self.SR = self.netG.super_resolution(
                    self.data, continous, cand=cand)

        # self.netG.train()

        return self.SR.detach().float().cpu()

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        # self.netG.train()
        return self.SR.detach().float().cpu()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self) -> OrderedDict:
        ''' Get the current log '''
        return self.log_dict

    def get_current_visuals(self, need_LR : bool = True, sample : bool = False):
        ''' Get the current visualization results 
        
        Parameters
        ----------
        need_LR : bool, optional
            Whether to return the low-resolution image. The default is True.
            
        sample : bool, optional
            Whether to return the sampled image. The default is False.
        '''
        # Create an ordered dictionary
        out_dict = OrderedDict()
        
        # If the sample flag is set, return the sampled image
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
            return out_dict          

        # Get the super-resolved image
        out_dict['SR'] = self.SR.detach().float().cpu()
        # Get the input image.
        out_dict['INF'] = self.data.detach().float().cpu()
        # out_dict['HR'] = self.data['HR'].detach().float().cpu()
        
        # If the low-resolution image is needed and it is present in the data, return it
        if need_LR and 'LR' in self.data:
            out_dict['LR'] = self.data.detach().float().cpu()
        # If the low-resolution image is not needed, return the input image
        else:
            out_dict['LR'] = out_dict['INF']
        return out_dict

    def print_network(self):
        ''' Print the network structure and the number of parameters '''
        # Get the network description
        s, n = self.get_network_description(self.netG)
        # If the network is a DataParallel object -- a model that is parallelized across multiple GPUs -- get the module
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(
                self.netG.__class__.__name__,
                self.netG.module.__class__.__name__
        )
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        # Print the network structure and the number of parameters
        self.logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        self.logger.info(s)

    def save_network(self, epoch : int, iter_step : int) -> None:
        ''' Save the model to the checkpoint directory'''
        gen_path = os.path.join(
            'checkpoint', 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            'checkpoint', 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                    'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)
        self.logger.info('Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self, finetune_norm : bool = False) -> None:
        ''' This function loads the pretrained models 
        
        Parameters
        ----------
        finetune_norm : bool, optional
            Whether to finetune the normalization layers. The default is False.
        '''
        # Load pretrained models
        if not self.weights:
            self.logger.info('Pretrained model not found')
            return
        
        # Loading pretrained model       
        self.logger.info('Loading pretrained model...'.format(self.weights))
        gen_path = '{}_gen.pth'.format(self.weights)
        
        # Generator model
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
            
        network.load_state_dict(torch.load(gen_path), strict=(not finetune_norm))

    def set_device(self, x : any) -> any:
        ''' Move the input to the device '''
        if isinstance(x, dict):
            # Iterate over the items in the dictionary
            for key, item in x.items():
                # If the item is not None
                if item is not None:
                    # Move the item to the device
                    x[key] = item.to(self.device)
        # If the input is a list
        elif isinstance(x, list):
            # Iterate over the items in the list
            for item in x:
                # If the item is not None
                if item is not None:
                    # Move the item to the device
                    item = item.to(self.device)
        # If the input is not a dictionary or a list
        else:
            # Move the input to the device
            x = x.to(self.device)
        # Return the input
        return x

    def get_network_description(self, network : any) -> tuple:
        '''Get the string and total parameters of the network
        
        Parameters
        ----------
        network : any
            The network.
        
        Returns
        -------
        tuple
            The string and total parameters of the network.
        '''
        # If the network is a DataParallel object -- a model that is parallelized across multiple GPUs -- get the module
        if isinstance(network, nn.DataParallel):
            network = network.module
        # Get the string representation of the network
        s = str(network)
        # Get the total number of parameters in the network
        n = sum(map(lambda x: x.numel(), network.parameters()))
        # Return the string and total parameters
        return s, n