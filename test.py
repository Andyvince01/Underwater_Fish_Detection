''' test.py
>>> 💥 This file is used to evaluate the performance of YOLO models
'''
#------------------------------------------------------------------------------#
# IMPORT LIBRARIES AND MODULES                                                 #    
#------------------------------------------------------------------------------#
import argparse, os
import cv2 as cv
import matplotlib.pyplot as plt

from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from utils import ParseKwargs, FISHSCALE_DIR

#------------------------------------------------------------------------------#
# FUNCTIONS                                                                    #
#------------------------------------------------------------------------------#

#--- FUNCTION: Test the model ---#
def test(model : str, weights : str, source : str, mode : str = None, **kwargs) -> None:
    '''Test the model on the given source. The source can be a directory (specified in the YAML file) or a single image.
    
    Parameters
    ----------
    model : str
        The model to test. The model is assumed to be in the 'ultralytics/cfg/models/v8/' directory.
    weights : str
        The weights of the model. The weights are assumed to be in the 'models/runs/detect/' directory.
    source : str
        The source to test the model on. The source can be a directory or a single image.
    mode : str
        The mode to test the model in. Default is test.
    kwargs : dict
        The keyword arguments to pass to the val method of the YOLO class.
    '''
    #--- Preliminaries ---#
    kwargs['project'] = os.path.join('models', 'results') if 'project' not in kwargs else kwargs['project']
    kwargs['name'] = model if 'name' not in kwargs else kwargs['name']
        
    #--- Load the the weights ---#
    yolo = YOLO(weights)
                
    # Test the model
    results = yolo.val(data=source, split=mode, **kwargs) if mode in ['train', 'test', 'val'] else yolo.predict(source=Image.open(source))
    
    #--- Draw the results on the image if mode is 'image' else return ---#
    if mode in ['train', 'test', 'val']: return
    
    draw_results(results, source)

#--- FUNCTION: Draw the results on the image ---#
def draw_results(results : list, source : str) -> None:
    ''' This function draws the results on the image.
    
    Parameters
    ----------
    results : list
        The results of the model.
    source : str
        The source to draw the results on.    
    '''    
    #--- Load the image ---#
    image = Image.open(source)
    
    #--- Define YOLO's Annotator ---#
    annotator = Annotator(image)
    
    #--- Draw the results on the image ---#
    for r in results:
        # Get the bounding box and the class
        boxes = r.boxes
        # Iterate over the bounding boxes
        for box in boxes:
            b = box.xyxy[0]         # Get the bounding box coordinates in the format (x1 = top-left x, y1 = top-left y, x2 = bottom-right x, y2 = bottom-right y)
            c = box.cls             # Get the class of the bounding box
            
            # Draw the bounding box
            annotator.box_label(b, label=c, color='red')
                
    #--- Save the image ---#
    new_image = annotator.result()
    Image.fromarray(new_image).save(f'{source.split(os.path.sep)[-1]}.jpg')

#------------------------------------------------------------------------------#
# MAIN                                                                         #
#------------------------------------------------------------------------------#
if __name__ == '__main__':
    #--- Parse the arguments from the command line ---#
    parser = argparse.ArgumentParser(description='Test the YOLO models on the FishScale dataset.')
    # Add the arguments
    parser.add_argument('--model', type=str, required=True, help='The path to the model to be tested.', default='YOLOv8s')
    parser.add_argument('-w', '--weights', type=str, help='The weights of the model. Default is the best weights.', default='best.pt')
    parser.add_argument('-s', '--source', type=str, help='The source to test the model on.')
    parser.add_argument('-i', '--image', type=str, help='The image to test the model on.')
    parser.add_argument('--mode', type=str, help='The mode to test the model in. Default is test.', choices=['image', 'train', 'test', 'val'])
    parser.add_argument(
        '-k', '--kwargs', 
        default={'verbose': True},
        nargs='*',
        action=ParseKwargs,
        help='The keyword arguments to pass to the val method of the YOLO class. The arguments should be in the format key1=value1, key2=value2,...',
    )

    # Parse the arguments
    args = parser.parse_args()
    
    #--- Modify the arguments ---#
    model = args.model if args.model is not None else 'YOLOv8'
    weights = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'runs', 'detect/', args.model, 'weights', args.weights)
    source = args.source if args.source is not None else args.image if args.image is not None else FISHSCALE_DIR
    mode = args.mode if args.mode is not None and args.image is not None else 'image' if args.image is not None else 'test'
    kwargs = args.kwargs if args.kwargs is not None else {}

    #--- Test the model ---#
    test(model=model, weights=weights, source=source, mode=mode, **kwargs)