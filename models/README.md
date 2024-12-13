# **🏋️ Fusion Process of Model Weights**

The **`misc.py`** includes Python utilities for merging and managing the weights of YOLO models. Specifically, it targets models from the YOLOv8 family, aiming to combine the weights of a pre-trained base model (e.g., `YOLOv8s`) with a model fine-tuned or trained from scratch for specialized tasks with different components (e.g., `YOLOv8s-FishScale`). 

The script takes two PyTorch state dictionaries as input:
1. **Base Model (`pt1`)**: Pre-trained on general datasets (e.g., `yolov8s.pt`).
2. **Custom Model (`pt2`)**: Trained on a specialized dataset or modified architecture (e.g., `yolov8s-FishScale_scratch.pt`).

The **`get_scores`** function determines offsets for re-mapping the layer keys based on the model's name or architecture. These offsets ensure proper alignment of weights during the merging process. For example models with `FishScale` layers may have specific index adjustments.

Using Python's **`re`** module, the keys in `pt1` are dynamically adjusted based on scores. A lambda function (`rematch`) applies the re-mapping logic by incrementing indices in the keys.

Then, the **`intersect_dicts`** function identifies keys common between the two models and ensures their corresponding tensors are compatible in shape. This avoids mismatches during weight merging.

Finally, the **`merge_pt`** function:
- Combines weights from the intersected keys.
- Preserves the unique structure and additional layers of the custom model (`pt2`).
- Saves the final fused model as a PyTorch file (e.g., `yolov8s-FishScale-full.pt`).

## Model Weights in the `weights/` Directory

The **`weights/`** directory serves as a repository for storing and organizing various pre-trained and fine-tuned weight files used in the development and testing of models based on the YOLOv8 architecture. Each weight file corresponds to a specific stage or variant of the training process.

- **`best_funiegan`** : Represents the best-performing model that integrates the FUnIE-GAN (Enhancement GAN for underwater images) into the YOLOv8s-FishScale architecture.
- **`best`**: Represents the best model trained on the `yolov8s-FishScale-full` weights.
- **`yolov8s-FishScale_scratch`**: Contains the weights of the YOLOv8s-FishScale model trained from scratch.
- **`yolov8s-FishScale-full`**: Represents the YOLOv8s-FishScale model obtained by merging: 
  - **Base weights (`yolov8s.pt`)**: Pre-trained on COCO for general object detection.
  - **Scratch weights (`yolov8s-FishScale_scratch`)**: Trained on a specific dataset for a specialized task.  
- **`yolov8s`**: Contains the best weights of the standard YOLOv8s model trained on the COCO dataset.