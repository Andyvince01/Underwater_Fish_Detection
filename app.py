''' app.py
>>> 💥 This file comprises the main function to run the demo of the project.
'''

#------------------------------------------------------------------------------#
# IMPORT LIBRARIES AND MODULES                                                 #
#------------------------------------------------------------------------------#
import cv2, os, spaces, uuid
import gradio as gr
import numpy as np

from collections import defaultdict
from pathlib import Path
from PIL import Image
from typing import Tuple, Union
from ultralytics import YOLO
from assets.ocean import Ocean

#--- Load the the weights ---#
yolo = YOLO('models/weights/best.pt')

def predict(source, conf_threshold, iou_threshold) -> Tuple[int, np.ndarray]:
    ''' This function predicts the results of the model on the given image.
    '''
    #--- Check if the source is a numpy array ---#
    if isinstance(source, np.ndarray): source = Image.fromarray(source)
    
    #--- Check if the source is a string ---#
    if isinstance(source, str): source = Image.open(source)
    
    #--- Check if the source is an image ---#
    if not isinstance(source, Image.Image): raise ValueError('The source must be an image.')
    
    #--- Predict the results ---#
    results = yolo.predict(source=source, conf=conf_threshold, iou=iou_threshold)
    
    #--- Draw the results on the image ---#
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

    return len(results), im

@spaces.GPU
def track(video, conf_threshold, iou_threshold):
        
    #--- Load the video ---#
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), "Error: Cannot open the video file!!!"
    
    #--- Set properties for the output video ---#
    video_codec = cv2.VideoWriter_fourcc(*"avc1")  # Using avc1 codec for better compatibility
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create temporary directory for videos if it doesn't exist
    temp_dir = "temp_videos"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    #--- Track history for each object ---#
    track_history = defaultdict(lambda: [])
    
    #--- Process each frame in the video ---#
    while cap.isOpened():
        #--- Read the next frame ---#
        ret, frame = cap.read()
        
        # Break if we've reached the end of the video
        if not ret: break
        
        #--- Run YOLO detection and tracking ---#
        results = yolo.track(source=frame, persist=True, conf=conf_threshold, iou=iou_threshold)

        #--- Extract detection results ---#
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        annotated_frame = results[0].plot()

        #--- Draw tracking lines for each detected object ---#
        if len(boxes) > 0:
            for box, track_id in zip(boxes, track_ids):
                x, y, *_ = box
                track = track_history[track_id]
                # Store center point of bounding box
                track.append((float(x), float(y)))
                # Keep only the last 30 points for the trail
                if len(track) > 30: track.pop(0)
                    
                # Draw the tracking trail
                points = np.array(track).reshape(-1, 1, 2).astype(np.int32)
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

        #--- Display the frame if no objects are detected ---#
        else:
            annotated_frame = frame
                    
        #--- Yield the annotated frame ---#
        yield (len(boxes), annotated_frame[:, :, ::-1])
        

#--- Create the interface ---#
theme = Ocean()

with gr.Blocks(
    css_paths=[Path("assets/ocean.css")],               # Load custom CSS file
    theme=theme                                         # Set the theme              
) as app:
    #--- Welcome screen ---#
    with gr.Group(visible=True) as welcome_screen:
        with gr.Column(elem_classes='welcome-container'):
            gr.HTML("""
                <video autoplay muted loop class="video-background">
                    <source src="https://videos.pexels.com/video-files/1918465/1918465-uhd_2560_1440_24fps.mp4" type="video/mp4">
                    Your browser does not support HTML5 video.
                </video>
            """)
            with gr.Column(elem_classes="content-overlay"):
                gr.HTML('<h1 class="welcome-title">YOLO-FishScale</h1>')
                gr.HTML('<p class="welcome-text">A real-time object detection and tracking system for marine life.</p>')
                start_btn = gr.Button("Start", elem_classes="start-button")
    
    #--- Main tabs ---#
    with gr.Tabs(visible=False) as main_tabs:
        #--- Image Detection Tab ---#
        with gr.TabItem("📸 Image Detection"):
            gr.Interface(
                fn=predict,
                inputs=[
                    gr.Image(type="pil", label="Upload Image"),                                 # Input image
                    gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),  # Confidence threshold
                    gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold")          # IoU threshold
                ],
                outputs=[
                    gr.Image(type='pil', label="Processed Image"),                              # Processed image
                    gr.Number(label="Fish Count", value=0, visible=True)         # Fish count
                ]
            )
                
        #--- Video Tracking Tab ---#
        with gr.TabItem("🎥 Video Tracking"):
            gr.HTML('<h1 class="video-title">Real-Time Video Tracking</h1>')
            gr.HTML('<p class="video-instructions">Upload a video to track marine life in real-time.</p>')
            gr.HTML('<p class="fish-counter"> Number of Fish Detected: <span id="fish-count">0</span></p>')
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Upload Video")
                    conf_slider = gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold")
                    iou_slider = gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold")
                
                with gr.Column():
                    processed_video = gr.Image(type='pil', label="Processed Video", streaming=True)
                    fish_counter = gr.Number(label="Fish Count", value=0, visible=True)
                    
            # Define the video tracking pipeline            
            video_input.change(
                fn=track,
                inputs=[video_input, conf_slider, iou_slider],
                outputs=[fish_counter, processed_video],
            )
                        
    start_btn.click(
        fn=lambda: (gr.Group(visible=False), gr.Tabs(visible=True)),
        outputs=[welcome_screen, main_tabs]
    )

if __name__ == "__main__":
    app.launch()