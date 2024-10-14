'''
> This script is used to test the model (DDPM) on a set of images.
'''
import logging, os, time, torch
from typing import Union
import model as Model
from utils import Logger, save_image, tensor2img

# Set the working directory
os.chdir('UIE-DM') if 'UIE-DM' not in os.getcwd() else None

# Initialize the results directory
os.makedirs('results', exist_ok=True)
results_directory = os.path.join('results', f"{len(os.listdir('results')):03d}")
os.makedirs(results_directory, exist_ok=True)

# Initialize the logger
logger = Logger(logger_name='Base', level=logging.INFO, log_filename=results_directory, screen=True)


def test(model : Model, source : Union[str, list]):
    ''' This function tests the model. 
    
    Parameters
    ----------
    model : Model (DDPM)
        The model to test.
    source : str | list
        The source of the image(s). If a string, it is the path of the image. If a list, it is a list of paths of the images.
    '''
    from PIL import Image
    from torchvision import transforms
    
    # If the source is a string, convert it to a list    
    if isinstance(source, str):
        source = [source]
    
    # Define the transformation to apply: convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),   # Converte in tensor e normalizza [0,1]
        transforms.Lambda(lambda x: x * 2 - 1)  # Porta il range da [0,1] a [-1,1]
    ])
    images = [transform(Image.open(os.path.join('data', img_path)).convert('RGB')) for img_path in source]
    
    enhanced_images = []
    try:
        for image in images:        
            # Feed the data to the model
            model.feed_data(image.unsqueeze(0))
            # Test the model
            start = time.time()
            enhanced_image = model.test(continous=False)
            print("Image type: ", type(enhanced_image))
            print("Image dtype: ", enhanced_image.dtype)
            end = time.time()
            print('Execution time:', (end - start), 'seconds')
            # Append the enhanced image to the list
            enhanced_images.append(enhanced_image)
    except torch.cuda.OutOfMemoryError:
        pass

    # Save the enhanced images
    for idx, enhanced_image in enumerate(enhanced_images):
        save_image(tensor2img(enhanced_image), os.path.join(results_directory, f"{idx:03d}.png"))        

if __name__ == "__main__":
    
    # Model Initialization
    diffusion = Model.create_model(phase='val')
    logger.info('Initial Model Finished')

    # Set the noise schedule
    diffusion.set_new_noise_schedule(
        schedule_opt={
            "schedule": "linear",
            "n_timestep": 2000,
            "linear_start": 1e-6,
            "linear_end": 1e-2
        }, 
        schedule_phase='val'
    )
    logger.info('Begin Model Inference.')

    # Test the model
    test(diffusion, ['26.jpg', '9908_Epinephelus_f000165.jpg', 'G000002_R.avi.58482.png', '280_png.rf.7a6f3fb622eb488d9e10d7b7dd9a21ca.jpg', '579_png.rf.725382a9a98810ea3f8d5c5e1bc3368c.jpg'])

    # # Stack delle immagini in un batch tensor (BxCxHxW)
    # images_tensor = torch.stack(images)
    
    # diffusion.feed_data(images_tensor)
    # start = time.time()
    # diffusion.test(continous=True)
    # end = time.time()
    
    # print('Execution time:', (end - start), 'seconds')
    
    # visuals = diffusion.get_current_visuals(need_LR=False, sample=True)
    # # hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
    # # fake_img = Metrics.tensor2img(visuals['SAM'])  # uint8
    # sr_img = Metrics.tensor2img(visuals['SAM'])  # uint8
    # # Metrics.save_img(
    # #     hr_img, '{}_{}_hr.png'.format(current_step, idx)
    # # )
    # # Metrics.save_img(
    # #     fake_img, '{}_{}_inf.png'.format(current_step, idx)
    # # )
    # Metrics.save_img(
    #     sr_img, '{}_{}_sr.png'.format(current_step, idx)
    # )
