from tkinter import Variable
from ultralytics.models import YOLO
import torch, os, cv2

import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
from funiegan import GeneratorFunieGAN
from torchvision.transforms import transforms
# Check if CUDA is available and set device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'

img_width, img_height, channels = 640, 640, 3
transforms_ = [
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]
transform = transforms.Compose(transforms_)

# Load the FunieGAN model
funiegan = GeneratorFunieGAN()
funiegan.load_state_dict(torch.load("funiegan/PyTorch/models/funie_generator.pth"))
funiegan.eval() 

def apply_funie_gan(image: str) -> torch.Tensor:
    
    def tensor_to_numpy(gen_img: torch.Tensor) -> np.ndarray:
        """
        Convert a PyTorch tensor to a numpy array suitable for visualization.
        
        Args:
            gen_img (torch.Tensor): The generated image tensor from the model.
            
        Returns:
            np.ndarray: The image in numpy array format.
        """
        # Remove the batch dimension (assuming the tensor has a shape like [1, C, H, W])
        gen_img = gen_img.squeeze(0)
        
        # Convert the tensor values from [-1, 1] to [0, 1]
        gen_img = gen_img.mul(0.5).add(0.5)
        
        # Convert the tensor values to [0, 255] for visualization
        gen_img = gen_img.mul(255.0)
        
        # Clamp values to ensure they are within [0, 255]
        gen_img = gen_img.clamp(0, 255)
        
        # Convert the tensor to uint8 type
        gen_img = gen_img.to(torch.uint8)
        
        # Move the tensor to the CPU
        gen_img = gen_img.cpu()
        
        # Convert the tensor to a numpy array
        gen_img_np = gen_img.numpy()
        
        return gen_img_np

    inp_img = transform(image).unsqueeze(0)
    with torch.no_grad():
        gen_img : torch.Tensor = funiegan(inp_img)
        
    # Convert the tensor to a numpy array and adjust the format
    gen_img_np = tensor_to_numpy(gen_img)
    
    # Display the image using Matplotlib
    gen_img_pil = Image.fromarray(gen_img_np)
    gen_img_pil.show()
    
    return gen_img

def main():  
        
    # Load the model
    model = YOLO("weights/yolov8n.pt")    
    
    # Train the model
    results = model.train(
        data="../data/fishscale_data.yaml", 
        epochs=50, 
        imgsz=640, 
        batch=16,
        device=device,
        workers=12,
        preproc_fn=apply_funie_gan,
    )
    
    # Save the trained model
    model.save("yolov8n_trained.pt")

if __name__ == '__main__':
    # Clear GPU cache
    torch.cuda.empty_cache()
    # Set the working directory
    os.chdir('models/') if os.getcwd().split("/")[-1] != "models" else None
    # Execute main function
    main()
    
    
# Ultralytics YOLOv8.2.68 🚀 Python-3.9.19 torch-2.3.0 CUDA:0 (NVIDIA GeForce RTX 3060 Laptop GPU, 6144MiB)
# Model summary (fused): 168 layers, 3,005,843 parameters, 0 gradients, 8.1 GFLOPs
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 48/48 [00:08<00:00,  5.46it/s]
#                    all       1523       9942      0.863      0.729      0.819      0.485
# Speed: 0.2ms preprocess, 1.8ms inference, 0.0ms loss, 0.7ms postprocess per image

# Model summary: 225 layers, 3,011,043 parameters, 3,011,027 gradients, 8.2 GFLOPs