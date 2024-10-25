''' models.misc.py
>>> 💥 This class
'''

#-------------------------------------------------------------------------------------------------#
# IMPORT MODULES AND/OR PACKAGES
#-------------------------------------------------------------------------------------------------#
import argparse, re, torch, ultralytics
from typing import OrderedDict

#-------------------------------------------------------------------------------------------------#
# FUNCTIONS
#-------------------------------------------------------------------------------------------------#
        
def intersect_dicts(da, db, exclude=()):
    """Returns a dictionary of intersecting keys with matching shapes, excluding 'exclude' keys, using da values."""
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}

def merge_pt(pt1 : OrderedDict, pt2 : OrderedDict) -> OrderedDict:
    ''' This function merges two weights into one 
    
    Parameters
    ----------
    pt1 : OrderedDict
        The first weight to merge. It is assumed that this weight is from the original model (e.g. yolov8s.pt)
    pt2 : OrderedDict
        The second weight to merge. It is assumed that this weight is from the model that was trained from scratch (e.g. yolov8s-FishScale_scratch.pt)
        
    Returns
    -------
    OrderedDict
        The merged weight of pt1 and pt2.
    '''    
    #--- Get the scores from the second weight ---#
    scores = get_scores(pt1=pt1, pt2=pt2)
    
    print(f"Scores: {scores}")
    
    #--- Define the lambda function to match the keys ---#
    rematch = lambda x: str(int(x.group(1)) + scores[int(x.group(1))])
    
    #--- Get the checkpoint state_dict as FP32 ---#
    ckpt1 = pt1["model"].model.float().state_dict()
    ckpt2 = pt2["model"].model.float().state_dict()

    #--- Reshape the keys of the first checkpoint ---#
    ckpt1 = {re.sub(r'(\d+)', rematch, k, count=1): v for k, v in ckpt1.items()}
    
    #--- Intersect the two weights ---#
    intersect = intersect_dicts(ckpt1, ckpt2)    

    #--- Merge the weights ---#
    state_dict_copy = pt2["model"].model.float().state_dict().copy()
    for key, value in intersect.items():
        state_dict_copy[key] = value

    pt2["model"].model.load_state_dict(state_dict_copy)
                                
    #--- Save the merged weight ---#
    torch.save(pt2, 'models/weights/yolov8s-FishScale-full.pt')

def get_scores(pt1 : OrderedDict, pt2 : OrderedDict) -> list:
    '''Get the scores from the second weight to add to the indices of the first weight.
    
    Parameters
    ----------
    pt1 : OrderedDict
        The state dictionary of the model. This is used to get the length of the scores if the model name is not found or it is YOLOv8 version.
    pt2 : OrderedDict
        The state dictionary of the model. This is used to get the scores that must be added to the indices of the first model.     

    Returns
    -------
    list
        The scores of indices corresponding to the new model's state dictionary.
    '''
    #--- Get the lenght of the first weight ---#
    length = len(list(pt1['model'].model.children()))
        
    #--- Get the model name ---#
    model_name = pt2['train_args']['name']

    #--- Return the model name based on the number of children ---#
    if "FishScale" in model_name:
        return [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 15, 16, 16, 17, 18, 18, 19]
    elif "p2-CBAM" in model_name:
        return
    elif "p2-SPD" in model_name:
        return [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 11, 12, 12, 12, 13, 13, 13]
    elif "p2" in model_name:
        return [0] * 16 + [6] * (length - 16)
    else:
        return [0] * length

#--- Main ---#
if __name__ == '__main__':
    #--- Define parser ---#
    parser = argparse.ArgumentParser(description='Merge the weights of two models.')
    # Add the arguments
    parser.add_argument('--model1', type=str, required=False, help='The path to the first model to be merged.', choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'], default='yolov8s')
    parser.add_argument('--model2', type=str, required=False, help='The path to the second model to be merged.', default='yolov8s-FishScale-full')
    # Parse the arguments
    args = parser.parse_args()

    #--- Load the two weights to merge ---#
    pt1 = torch.load(f'models/weights/{args.model1}.pt')
    pt2 = torch.load(f'models/runs/detect/{args.model2}/weights/best.pt')
    ckpt2 = {k: v.clone() for k, v in pt2['model'].model.float().state_dict().items()}
        
    #--- Merge the two weights ---#
    # merge_pt(pt1, pt2)
        
    pt3 = torch.load('models/weights/yolov8s-FishScale-full.pt')
    
    scores = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 15, 16, 16, 17, 18, 18, 19]
    rematch = lambda x: str(int(x.group(1)) + scores[int(x.group(1))])
    ckpt1 = {re.sub(r'(\d+)', rematch, k, count=1): v for k, v in pt1['model'].model.float().state_dict().items()}
    
    # Save on text file the keys of the two weights
    with open(f'FISHSCALE.txt', 'w') as f:
        f.write("===YOLOV8S KEYS===\n")
        f.write(str(pt1['model'].model.float().state_dict().items()))
        f.write('\n\n')
        f.write("===YOLOV8S-FISHSCALE FROM SCRATCH KEYS===\n")
        f.write(str(ckpt2.items()))
        f.write('\n\n')
        f.write("===YOLOV8S-FISHSCALE KEYS===\n")
        f.write(str(pt3['model'].model.float().state_dict().items()))
        f.write('\n\n')
        
        f.write("===MATCHING KEYS===\n")
        for k3, v3 in pt3['model'].model.float().state_dict().items():
            try:
                v1 = ckpt1[k3]
            except:
                v1 = None
            v2 = ckpt2[k3]
            if torch.allclose(v2, v3, atol=1e-05):
                f.write(f"YOLOv8s-FISHSCALE - {k3} : v1 = {v1.shape if v1 is not None else 'None'} | v2 = {v2.shape if v2 is not None else 'None'}\n")
            elif torch.allclose(v1, v3, atol=1e-05):
                f.write(f"YOLOV8S - {k3} : v1 = {v1.shape if v1 is not None else 'None'} | v2 = {v2.shape if v2 is not None else 'None'}\n")
            else:
                f.write("ERROR!!!\n")
                