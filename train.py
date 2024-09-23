'''
    > This file is used to train the different models on the fisheye-scale dataset.
'''

import argparse, json, os
from ultralytics import YOLO, YOLOGAN

from utils import ParseKwargs, FISHSCALE_DIR

# Define the directories for the fisheye-scale dataset and the weights

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/'))

def train(model : str = 'yolov8n', weights : str = None, verbose : bool = True, **kwargs : dict):
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
    # Load the model and the weights
    yolo = YOLO(model=model).load(weights)
    
    print("\n=== The model is loaded ===\n") if verbose else None    
    
    # Train the model
    yolo.train(data=FISHSCALE_DIR, **kwargs)

# Run the script
if __name__ == '__main__':
    # Define the command line arguments
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
        # choices=[
        #     '', 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
        # ]
    )
    parser.add_argument(
        '--kwargs',
        default={'batch': 32, 'epochs': 100, 'patience': 35, 'workers': 8},
        nargs='*',
        action=ParseKwargs,
        help="Override the default settings for the model. The settings are passed as a dictionary. For example: --kwargs batch=64 epochs=100 ...}."
    )
    # Parse the arguments
    args = parser.parse_args()
    
    # Get the model file
    model = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ultralytics', 'cfg', 'models', 'v8', args.model + '.yaml')

    # Get the weights file
    weights = args.weights + ".pt" if args.weights and ".pt" not in args.weights else None

    # Get extra keyword arguments
    kwargs = args.kwargs

    # Train the model
    train(model, weights, **kwargs) 
    
    ''' (NO FUNIEGAN)
        
        ------------------------- YOLOV8 TRAINING -------------------------
        Ultralytics YOLOv8.2.79 🚀 Python-3.8.10 torch-2.2.1+cu121 CUDA:0 (NVIDIA A100-SXM4-40GB MIG 1g.10gb, 9856MiB)
        YOLOv8n summary (fused): 168 layers, 3,005,843 parameters, 0 gradients
                        Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 112/112 [00:10<00:00, 10.61it/s]
                        all       1779      11045      0.875      0.718       0.81      0.482
        Speed: 0.2ms preprocess, 2.8ms inference, 0.0ms loss, 0.5ms postprocess per image
        --- (No Pretrained Weights) ---
        Validating /user/aricciardi/Underwater Fish Detection/Underwater_Fish_Detection/runs/detect/train/weights/best.pt...
        Ultralytics YOLOv8.2.79 🚀 Python-3.8.10 torch-2.2.1+cu121 CUDA:0 (NVIDIA A100-SXM4-40GB MIG 1g.10gb, 9856MiB)
        YOLOv8n summary (fused): 168 layers, 3,005,843 parameters, 0 gradients
                        Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 112/112 [00:10<00:00, 10.60it/s]
                        all       1779      11045       0.86      0.678      0.769      0.443
        Speed: 0.2ms preprocess, 2.8ms inference, 0.0ms loss, 0.6ms postprocess per image  
        
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
    '''