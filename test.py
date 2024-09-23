''' test.py
>   This file is used to test the functionality of the YOLO models on a sample image or a given test set.
'''

import os, cv2 as cv
import matplotlib.pyplot as plt

from ultralytics import YOLO, YOLOGAN
from ultralytics.utils.plotting import Annotator
from utils import FISHSCALE_DIR

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/'))
os.makedirs('results', exist_ok=True)

def test_on_set(source, model) -> None:
    '''Test the model on the given source.

    Parameters
    ----------
    source : str
        The source to test the model on. The source can be a single image or a directory of images.
    model : str
        The model to test. The model is assumed to be in the 'ultralytics/cfg/models/v8/' directory.
    '''
    # Get the model name and create the directory for the results, if it does not exist
    model_name = model.split('.')[0]
    os.makedirs(f'results/{model_name}', exist_ok=True)    
    
    # Load the model and the weights
    yolo = YOLO(model=model)
            
    # Test the model
    yolo.val(data=FISHSCALE_DIR, split='test', verbose = True)
    
def test_on_image(source : str, model : str) -> None:
    '''Test the model on the given image.

    Parameters
    ----------
    source : str
        The source to test the model on. The source is assumed to be a single image.
    model : str
        The model to test. The model is assumed to be in the 'ultralytics/cfg/models/v8/' directory.
    '''
    
    # Read the image
    source = cv.imread(source, cv.IMREAD_COLOR)
    
    # Load the model and the weights
    yolo = YOLO(model=model)
    
    # Test the model on the image
    results = yolo.predict(source)
    
    # Draw the results on the image
    annotator = Annotator(source)
    
    for r in results:
        # Get the bounding box and the class
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]         # Get the bounding box coordinates in the format (x1 = top-left x, y1 = top-left y, x2 = bottom-right x, y2 = bottom-right y)
            c = box.cls             # Get the class of the bounding box
            
            # Draw the bounding box
            annotator.box_label(b, label=c, color='red', line_thickness=2)
        
    image = annotator.result()
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.show()
    
if __name__ == '__main__':
    # Test the model on a single image    
    # test_on_image(
    #     source=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/fishscale_dataset/train/images/0_png.rf.6710c020ead3bd2aa5b863cd55532c0d.jpg'),
    #     model=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs/detect/train/weights/best.pt')
    # )
    
    test_on_set(
        source=FISHSCALE_DIR,
        model=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs/detect/YOLOv8s-p2/weights/best.pt')
    )
        