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
        ]
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='',
        help='The weights to load for the model. If not specified, the model is trained from scratch.',
        choices=['', 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
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
    weights = args.weights + ".pt" if args.weights and ".pt" not in args.weights else None
    kwargs = args.kwargs if args.kwargs is not None else {}

    #--- Train the model ---#
    train(model=model, weights=weights, **kwargs) 
    
    ''' (NO FUNIEGAN)
        
        ------------------------- YOLOV8 TRAINING -------------------------
        Ultralytics YOLOv8.2.79 🚀 Python-3.8.10 torch-2.2.1+cu121 CUDA:0 (NVIDIA A100-SXM4-40GB MIG 1g.10gb, 9856MiB)
        YOLOv8n summary (fused): 168 layers, 3,005,843 parameters, 0 gradients
                        Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 112/112 [00:10<00:00, 10.61it/s]
                        all       1779      11045      0.875      0.718       0.81      0.482
        Speed: 0.2ms preprocess, 2.8ms inference, 0.0ms loss, 0.5ms postprocess per image
        --- (No Pretrained Weights) ---
        Validating /user/aricciardi/Underwater_Fish_Detection/runs/detect/YOLOv8s-p2 from scratch/weights/best.pt...
        Ultralytics YOLOv8.2.79 🚀 Python-3.8.10 torch-2.2.2+cu121 CUDA:0 (NVIDIA L40S, 45495MiB)
        YOLOv8s summary (fused): 168 layers, 11,125,971 parameters, 0 gradients, 28.4 GFLOPs
                        Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 56/56 [00:07<00:00,  7.75it/s]
                        all       1779      11045      0.856      0.717      0.827      0.508
        Speed: 0.1ms preprocess, 0.7ms inference, 0.0ms loss, 0.5ms postprocess per image
                
        ------------------------- YOLOV8-P2 TRAINING -------------------------       
        Ultralytics YOLOv8.2.79 🚀 Python-3.8.10 torch-2.2.1+cu121 CUDA:0 (NVIDIA A100-SXM4-40GB MIG 1g.10gb, 9856MiB)
        YOLOv8n-p2 summary (fused): 207 layers, 2,921,172 parameters, 0 gradients
                        Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 112/112 [00:13<00:00,  8.10it/s]
                        all       1779      11045      0.876       0.72      0.814      0.481
        Speed: 0.2ms preprocess, 4.7ms inference, 0.0ms loss, 0.6ms postprocess per image
        --- (No Pretrained Weights) ---
        Validating /user/aricciardi/Underwater Fish Detection/Underwater_Fish_Detection/runs/detect/train/weights/best.pt...
        Ultralytics YOLOv8.2.79 🚀 Python-3.8.10 torch-2.2.1+cu121 CUDA:0 (NVIDIA A100-SXM4-40GB MIG 1g.10gb, 9856MiB)
        YOLOv8n-p2 summary (fused): 207 layers, 2,921,172 parameters, 0 gradients
                        Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 112/112 [00:14<00:00,  7.85it/s]
                        all       1779      11045       0.86        0.7      0.795      0.463
        Speed: 0.2ms preprocess, 4.7ms inference, 0.0ms loss, 0.6ms postprocess per image
        
        ------------------------- YOLOV8-P2-SPD TRAINING -------------------------
        Ultralytics YOLOv8.2.79 🚀 Python-3.8.10 torch-2.2.1+cu121 CUDA:0 (NVIDIA A100-SXM4-40GB MIG 1g.10gb, 9856MiB)
        YOLOv8n-p2-SPD summary (fused): 214 layers, 3,311,316 parameters, 0 gradients
                        Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 112/112 [00:16<00:00,  6.86it/s]
                        all       1779      11045      0.849      0.717      0.804      0.467
        Speed: 0.2ms preprocess, 6.0ms inference, 0.0ms loss, 0.7ms postprocess per image
        --- (No Pretrained Weights) ---
        Validating /user/aricciardi/Underwater_Fish_Detection/runs/detect/YOLOv8s-p2-SPD from scratch/weights/best.pt...
        Ultralytics YOLOv8.2.79 🚀 Python-3.8.10 torch-2.2.2+cu121 CUDA:0 (NVIDIA L40S, 45495MiB)
        YOLOv8s-p2-SPD summary (fused): 214 layers, 12,187,284 parameters, 0 gradients, 55.7 GFLOPs
                        Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 56/56 [00:09<00:00,  5.65it/s]
                        all       1779      11045      0.852      0.734      0.814      0.473
        Speed: 0.1ms preprocess, 1.8ms inference, 0.0ms loss, 0.9ms postprocess per image
    
        Ultralytics YOLOv8.2.79 🚀 Python-3.8.10 torch-2.2.2+cu121 CUDA:0 (NVIDIA L40S, 45495MiB)
        YOLOv8s-FishScale summary: 328 layers, 17,358,240 parameters, 0 gradients, 70.0 GFLOPs
                        Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 56/56 [00:10<00:00,  5.24it/s]
                        all       1779      11045      0.862      0.736      0.841      0.529
        Speed: 0.1ms preprocess, 2.2ms inference, 0.0ms loss, 0.9ms postprocess per image
        Results saved to /user/aricciardi/Underwater_Fish_Detection/runs/detect/YOLOv8n-FishScale from scratch
        
        Validating /user/aricciardi/Underwater_Fish_Detection/runs/detect/YOLOv8s-FishScale from scratch3/weights/best.pt...
        Ultralytics YOLOv8.2.79 🚀 Python-3.8.10 torch-2.2.2+cu121 CUDA:0 (NVIDIA L40S, 45495MiB)
        YOLOv8s-FishScale summary: 328 layers, 17,358,240 parameters, 0 gradients, 70.0 GFLOPs
                        Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 56/56 [00:10<00:00,  5.43it/s]
                        all       1779      11045      0.863      0.739      0.843      0.533
        Speed: 0.1ms preprocess, 2.2ms inference, 0.0ms loss, 0.7ms postprocess per image
        Results saved to /user/aricciardi/Underwater_Fish_Detection/runs/detect/YOLOv8s-FishScale from scratch3
    '''
    '''NEW
    
        Ultralytics YOLOv8.2.79 🚀 Python-3.8.10 torch-2.2.2+cu121 CUDA:0 (NVIDIA L40S, 45495MiB)
        YOLOv8s summary (fused): 168 layers, 11,125,971 parameters, 0 gradients, 28.4 GFLOPs
                        Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 56/56 [00:08<00:00,  6.86it/s]
                        all       1779      11045      0.876      0.739      0.848      0.536
        Speed: 0.1ms preprocess, 0.8ms inference, 0.0ms loss, 0.6ms postprocess per image
        Results saved to /user/aricciardi/Underwater_Fish_Detection/runs/detect/YOLOv8s
        Ultralytics YOLOv8.2.79 🚀 Python-3.8.10 torch-2.2.2+cu121 CUDA:0 (NVIDIA L40S, 45495MiB)
        
        YOLOv8s-p2 summary (fused): 207 layers, 10,626,708 parameters, 0 gradients, 36.6 GFLOPs
                        Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 56/56 [00:08<00:00,  6.33it/s]
                        all       1779      11045      0.874      0.744      0.851      0.536
        Speed: 0.1ms preprocess, 1.4ms inference, 0.0ms loss, 0.8ms postprocess per image
        Results saved to /user/aricciardi/Underwater_Fish_Detection/runs/detect/YOLOv8s-p2
        
        Ultralytics YOLOv8.2.79 🚀 Python-3.8.10 torch-2.2.2+cu121 CUDA:0 (NVIDIA L40S, 45495MiB)
        YOLOv8s-p2-SPD summary (fused): 214 layers, 12,187,284 parameters, 0 gradients, 55.7 GFLOPs
                        Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 56/56 [00:09<00:00,  5.75it/s]
                        all       1779      11045       0.87      0.742      0.847      0.536
        Speed: 0.1ms preprocess, 1.9ms inference, 0.0ms loss, 0.5ms postprocess per image
        Results saved to /user/aricciardi/Underwater_Fish_Detection/runs/detect/YOLOv8s-p2-SPD
        
        Ultralytics YOLOv8.2.79 🚀 Python-3.8.10 torch-2.2.2+cu121 CUDA:0 (NVIDIA L40S, 45495MiB)
        YOLOv8s-FishScale summary: 328 layers, 17,358,240 parameters, 0 gradients, 70.0 GFLOPs
                        Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 56/56 [00:10<00:00,  5.28it/s]
                        all       1779      11045      0.867      0.748       0.85      0.541
        Speed: 0.1ms preprocess, 2.2ms inference, 0.0ms loss, 0.8ms postprocess per image
    '''