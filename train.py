'''
    > This file is used to train the different models on the fisheye-scale dataset.
'''

import argparse, os
from ultralytics import YOLO

# Define the directories for the fisheye-scale dataset and the weights
FISHSCALE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'fishscale_data.yaml')
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'weights')

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/'))

def train(model : str = 'yolov8', weights : str = 'yolov8n.pt', **settings):
    '''Train the model on the fisheye-scale dataset.

    Parameters
    ----------
    model : str
        The model to train. The model is assumed to be in the 'ultralytics/cfg/models/v8/' directory.
        By default, the model is set to 'yolov8'.
    weights : str
        The weights to use for the model. The weights are assumed to be in the 'models' directory.
        By default, the weights are set to 'yolov8n.pt'.
    **settings : dict
        The settings for the model. The settings are passed as a dictionary.
        For example: {'batch_size': 64, 'epochs': 100, 'learning_rate': 0.001}.
    '''
    # Load the model and the weights
    yolo = YOLO(model).load(weights)

    # Train the model
    yolo.train(data=FISHSCALE_DIR, **settings)

    # Save the trained model
    yolo.save(os.path.join(WEIGHTS_DIR, f'{model}_trained.pt'))

if __name__ == '__main__':
    # Define the command line arguments
    parser = argparse.ArgumentParser(description='Train a YOLO model on the fisheye-scale dataset.')
    # Add the arguments to the parser
    parser.add_argument(
        '--model', 
        type=str, 
        default='yolov8n-p2-SPD', 
        help='The model to train (default: yolov8n). It is assumed that the model is in \'ultralytics/cfg/models/v8/\' directory.', 
        choices=['yolov8', 'yolov8-p2', 'yolov8-p2-SPD', 'yolov8-p2-CBAM', 'yolov8-FishScale']
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='yolov8n.pt',
        help='The weights to use for the model (default: yolov8n.pt). It is assumed that the weights are in the \'models\' directory.',
        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
    )
    parser.add_argument(
        '--settings',
        type=dict,
        default={},
        help="Override the default settings for the model. The settings are passed as a dictionary. For example: {'batch_size': 64, 'epochs': 100, 'learning_rate': 0.001}."
    )
    # Parse the arguments
    args = parser.parse_args()
    
    # Get the model file
    model = os.path.join(os.getcwd(), '..\\ultralytics', 'cfg', 'models', 'v8', args.model + '.yaml')

    # Get the weights file
    weights = args.weights
    
    # Get extra settings
    settings = args.settings
    
    # Train the model
    train(model, weights, **settings) 