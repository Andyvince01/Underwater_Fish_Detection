# 🐟 **Fishscale Dataset**

Welcome to the **Fishscale Dataset** 🐠! This dataset has been crafted to support the training and evaluation of fish detection models using `YOLOv8`. Here's a detailed overview of what you need to know about it.

## 📜 **Dataset Overview**

The Fishscale Dataset is a comprehensive collection of images and labels derived from three well-known fish datasets:

- **Deepfish** [1]: This dataset comprises footage from 20 distinct underwater habitats, offering a diverse representation of underwater environments. This dataset was not originally intended for object detection tasks, which means it did not include bounding box annotations for fish. In the context of this project, the same dataset used by A.A. Muksit et al. [4] was utilized. To adapt the DeepFish dataset for fish detection, A.A. Muksit et al. [4] manually curated a subset of images, selecting those that exhibited different types of fish movement and posture across various habitats. From this selection process, they identified 4,505 positive images and then meticulously annotated a total of 15,463 ground truth bounding boxes within these images.
- **Fish4knowledge from Roboflow** [2]: This dataset is derived from the larger Fish4Knowledge dataset and includes 1,879 frames that have been pre-processed and annotated specifically for use with the YOLOv8 format.
- **Ozfish** [3]: This dataset was developed as part of the Australian Research Data Commons Data Discoveries program to support research in machine learning for automated fish detection from video footage. It contains approximately 43,000 bounding box annotations across nearly 1,800 frames, making it a substantial resource for object detection tasks. What sets the OzFish dataset apart is its diversity; each image includes multiple species and various shapes of fish, with an average of 25 fish per frame. Some frames feature between 80 and 120 fish, presenting a significant challenge due to the high density of fish objects, many of which are small and difficult to detect. In contrast, the DeepFish dataset typically has 3 to 4 fish per frame, with a maximum of about 14 in some cases. The prevalence of numerous tiny fish in OzFish makes it particularly challenging for detection algorithms, especially when it comes to accurately identifying and differentiating among the small, densely packed objects.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b54003f7-0946-4214-aee2-a6c8778446c0" width="33%" alt="Image 1">
  <img src="https://github.com/user-attachments/assets/25898ac6-df24-46d6-9423-4ff4b33fd21e" width="33%" alt="Image 2">
  <img src="https://github.com/user-attachments/assets/89a04c25-31da-4cdf-a747-334da708011f" width="33%" alt="Image 3">
</p>

<p align="center">
  <strong>Figure 1</strong>: Example images from the Fishscale Dataset. The image on the left is from the DeepFish dataset, the middle image is from the Fish4Knowledge dataset, and the image on the right is from the OzFish dataset.
</p>

To use this dataset, clone or download the repository, and then run the `fishscale_dataset_generator.py` script to generate the dataset. Make sure that you have the required source datasets placed in their respective directories. Alternatively, you can download the dataset directly from the following [link](https://drive.google.com/file/d/1QLFlUPOqu-xgKdRceyP9dNeUGyXRBFbL/view?usp=sharing), which includes the test set used by A. A. Muksit [4].

## 📂 **Dataset Structure**

The dataset comprises a total of **10,154** images and their corresponding labels. It is meticulously organized into three subsets: _Training_, _Validation_, and _Test_.

- **`fishscale_dataset/train/`**  
  - **`images/`**: Training images.
  - **`labels/`**: Labels for training images.

- **`fishscale_dataset/valid/`**  
  - **`images/`**: Validation images.
  - **`labels/`**: Labels for validation images.

- **`fishscale_dataset/test/`**  
  - **`images/`**: Test images.
  - **`labels/`**: Labels for test images.

## ⚙️ **Generating the Dataset** 

The dataset was created using the `fishscale_dataset_generator.py` script. This script performs the following tasks:

1. **Copying Data**: Copies images and labels from the **Deepfish** ([link](https://drive.google.com/file/d/10Pr4lLeSGTfkjA40ReGSC8H3a9onfMZ0/view?usp=sharing)), **Fish4Knowledge** ([link](https://universe.roboflow.com/g18l5754/fish4knowledge-dataset)), and **Ozfish** ([link](https://github.com/open-AIMS/ozfish)) datasets into the new structure.
2. **Splitting Data**: Divides the dataset into training, validation, and test sets. You can specify whether to use a fixed test set or generate a random one.

For a detailed explanation of the script and its functions, refer to the code comments and documentation within `fishscale_dataset_generator.py`.

## 📋 **References**

For detailed information about the dataset and its usage, please refer to:

[1] Saleh, Alzayat, Laradji, Issam H., Konovalov, Dmitry A., Bradley, Michael, Vazquez, David, Sheaves, Marcus, 2020. A realistic fish-habitat dataset to evaluate algorithms for underwater visual analysis. Sci. Rep. 10 (1), 1–10. [doi:10.53654/tangible.v5i1.110](https://doi.org/10.53654/tangible.v5i1.110).

[2] Fish4Knowledge Dataset. g18L5754. Fish4Knowledge Dataset. Open Source Dataset. Roboflow Universe, October 2023. Available at: https://universe.roboflow.com/g18l5754/fish4knowledge-dataset. 

[3] Australian Institute Of Marine Science, 2020. Ozfish dataset - machine learning dataset for baited remote underwater video stations. 

[4] A. A. Muksit, F. Hasan, M. F. Hasan Bhuiyan Emon, M. R. Haque, A. R. Anwary, and S. Shatabda, “Yolo-fish: A robust fish detection model to detect fish in realistic underwater environment,” Ecological Informatics, vol. 72, p. 101847, 2022. [doi:10.1016/j.ecoinf.2022.101847](https://doi.org/10.1016/j.ecoinf.2022.101847).


**Happy training! 🎣**
