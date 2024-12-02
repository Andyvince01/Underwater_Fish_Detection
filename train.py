''' train.py
>>> 💥 This file is used to train YOLO models.
'''
#------------------------------------------------------------------------------#
# IMPORT LIBRARIES AND MODULES                                                 #
#------------------------------------------------------------------------------#
import argparse, os

from ultralytics import YOLO
from utils import ParseKwargs, FISHSCALE_DIR

#------------------------------------------------------------------------------#
# FUNCTIONS                                                                    #
#------------------------------------------------------------------------------#

#--- FUNCTION: Train the model ---#
def train(model : str = 'yolov8n', weights : str = None, **kwargs : dict):
    '''Train the model on the fisheye-scale dataset.

    Parameters
    ----------
    model : str
        The model to train. The model is assumed to be in the 'ultralytics/cfg/models/v8/' directory.
        By default, the model is set to 'yolov8n'.
    weights : str
        The weights to load for the model. If not specified, the model is trained from scratch.
        By default, the weights are set to an empty string (scratch training).
    **kwargs : dict
        The kwargs for the model. The kwargs are passed as a dictionary.
        For example: {'batch': 64, 'epochs': 100, 'learning_rate': 0.001}.
    '''
    #--- Preliminaries ---#
    kwargs['project'] = os.path.join('models', 'runs', 'detect') if 'project' not in kwargs else kwargs['project']
    kwargs['name'] = model.split('v')[0].upper() + 'v' + model.split('v')[1] + ('' if weights else ' from scratch') if 'name' not in kwargs else kwargs['name']
    model = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ultralytics', 'cfg', 'models', 'v8', args.model + '.yaml')
    weights = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'weights', weights) if weights else None

    #--- Load the model ---#
    yolo = YOLO(model=model).load(weights)

    #--- Train the model ---#    
    yolo.train(data=FISHSCALE_DIR, **kwargs)

#------------------------------------------------------------------------------#
# MAIN                                                                         #
#------------------------------------------------------------------------------#
if __name__ == '__main__':
    #--- Parse the arguments from the command line ---#
    parser = argparse.ArgumentParser(description='Train a YOLO model on the fisheye-scale dataset.')
    # Add the arguments to the parser
    parser.add_argument(
        '--model', 
        type=str, 
        default='yolov8n', 
        help='The model to train (default: yolov8n). It is assumed that the model is in \'ultralytics/cfg/models/v8/\' directory.', 
        choices=[
            'yolov8n', 'yolov8n-p2', 'yolov8n-p2-SPD', 'yolov8n-p2-CBAM', 'yolov8n-FishScale',
            'yolov8s', 'yolov8s-p2', 'yolov8s-p2-SPD', 'yolov8s-p2-CBAM', 'yolov8s-FishScale',
            'yolov8m', 'yolov8m-p2', 'yolov8m-p2-SPD', 'yolov8m-p2-CBAM', 'yolov8m-FishScale',
            'yolov8l', 'yolov8l-p2', 'yolov8l-p2-SPD', 'yolov8l-p2-CBAM', 'yolov8l-FishScale',
            'yolov8x', 'yolov8x-p2', 'yolov8x-p2-SPD', 'yolov8x-p2-CBAM', 'yolov8x-FishScale',
            'yolov8n_FunieGAN', 'yolov8n-p2_FunieGAN', 'yolov8n-p2-SPD_FunieGAN', 'yolov8n-p2-CBAM_FunieGAN', 'yolov8n-FishScale_FunieGAN',
            'yolov8s_FunieGAN', 'yolov8s-p2_FunieGAN', 'yolov8s-p2-SPD_FunieGAN', 'yolov8s-p2-CBAM_FunieGAN', 'yolov8s-FishScale_FunieGAN',
            'yolov8m_FunieGAN', 'yolov8m-p2_FunieGAN', 'yolov8m-p2-SPD_FunieGAN', 'yolov8m-p2-CBAM_FunieGAN', 'yolov8m-FishScale_FunieGAN',
            'yolov8l_FunieGAN', 'yolov8l-p2_FunieGAN', 'yolov8l-p2-SPD_FunieGAN', 'yolov8l-p2-CBAM_FunieGAN', 'yolov8l-FishScale_FunieGAN',
            'yolov8x_FunieGAN', 'yolov8x-p2_FunieGAN', 'yolov8x-p2-SPD_FunieGAN', 'yolov8x-p2-CBAM_FunieGAN', 'yolov8x-FishScale_FunieGAN',
            'yolov8n_UIEDM', 'yolov8n-p2_UIEDM', 'yolov8n-p2-SPD_UIEDM', 'yolov8n-p2-CBAM_UIEDM', 'yolov8n-FishScale_UIEDM',
            'yolov8s_UIEDM', 'yolov8s-p2_UIEDM', 'yolov8s-p2-SPD_UIEDM', 'yolov8s-p2-CBAM_UIEDM', 'yolov8s-FishScale_UIEDM',
            'yolov8m_UIEDM', 'yolov8m-p2_UIEDM', 'yolov8m-p2-SPD_UIEDM', 'yolov8m-p2-CBAM_UIEDM', 'yolov8m-FishScale_UIEDM',
            'yolov8l_UIEDM', 'yolov8l-p2_UIEDM', 'yolov8l-p2-SPD_UIEDM', 'yolov8l-p2-CBAM_UIEDM', 'yolov8l-FishScale_UIEDM',
            'yolov8x_UIEDM', 'yolov8x-p2_UIEDM', 'yolov8x-p2-SPD_UIEDM', 'yolov8x-p2-CBAM_UIEDM', 'yolov8x-FishScale_UIEDM',
        ]
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='',
        help='The weights to load for the model. If not specified, the model is trained from scratch.',
    )
    parser.add_argument(
        '--kwargs',
        default={'batch': 32, 'epochs': 100, 'patience': 35, 'workers': 2},
        nargs='*',
        action=ParseKwargs,
        help="Override the default settings for the model. The settings are passed as a dictionary. For example: --kwargs batch=64 epochs=100 ...}."
    )
    # Parse the arguments
    args = parser.parse_args()
    
    #--- Modify the arguments ---#
    model = args.model if args.model else 'yolov8n'
    weights = args.weights + (".pt" if '.pt' not in args.weights else '') if args.weights else None
    kwargs = args.kwargs if args.kwargs is not None else {}

    #--- Train the model ---#
    train(model=model, weights=weights, **kwargs) 