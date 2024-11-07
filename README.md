# 🐠 YOLO-FishScale - Underwater Fish Detection

<div align="center">
  <a href='https://huggingface.co/spaces/Andyvince01/Underwater_Fish_Detection'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-HF-yellow'></a> &ensp;
  <a href="https://andyvince01.github.io/YOLO-FishScale/"><img src="https://img.shields.io/static/v1?label=Official Website&message=Page&color=blue"></a> &ensp;
</div>

Freshwater ecosystems are facing significant challenges due to the extinction of endemic species, often caused by the invasion of aggressive species in degraded habitats. Overfishing and unsustainable practices are further threatening fish populations, destabilizing aquatic environments. This crisis is rooted in the exponential growth of the human population, which intensifies environmental degradation. To address declining wild fish stocks, aquaculture has emerged as a vital solution, not only for food production but also for environmental conservation by restoring natural habitats and replenishing wild populations. In this context, deep learning techniques are revolutionizing aquaculture by enabling precise monitoring and management of aquatic environments. The ability to process and analyze large volumes of visual data in real-time helps in accurately detecting, tracking, and understanding fish behavior, which is crucial for both optimizing aquaculture practices and preserving natural ecosystems.

This thesis presents a real-time system for detecting and tracking fish in underwater environments, utilizing a custom fish detector called **`YOLO-FishScale`** based on the `YOLOv8` algorithm. This detector addresses the challenge of detecting small fish, which vary in size and distance within frames. It enhances YOLOv8 by adding a new detection head that uses features from the $P_2$ layer and replaces the Conv module with `SPD-Conv`, improving performance on small and low-resolution targets. Additionally, `CBAM` (Convolutional Block Attention Mechanism) modules are integrated to enhance feature fusion, resulting in more accurate fish detection and tracking.

![image](https://github.com/user-attachments/assets/2782c6c2-32dd-4a16-8b21-f3bd701e0d03)


>[!NOTE]
>The original YOLOv8 model employs a backbone network that down-samples the image through five stages, resulting in five feature layers ($P_1$, $P_2$, $P_3$, $P_4$, and $P_5$). Here, each $P_i$ layer represents a resolution of $1/2^i$ of the original image. To address the challenge of detecting small fish more effectively, `YOLO-FishScale` proposes the addition of a new detection head to the `YOLOv8` architecture. This additional detection head utilizes features from the $P_2$ _layer_, which is specifically designed to enhance micro-target detection capabilities. Small objects are particularly challenging to detect due to their low resolution, which provides limited information about the content needed to learn patterns.

![ezgif-1-a1147faba4](https://github.com/user-attachments/assets/f452674b-8533-4c17-9989-95ae326504ed)
![ezgif-1-a8ae202b0a](https://github.com/user-attachments/assets/4c821c68-dcb8-4479-99f8-51ec85ce88b5)

## 🎣 FishScale Dataset
To train the `YOLO-FishScale` model,, a new comprehensive dataset was constructed. The **Fishscale Dataset** is an extensive collection of images and labels, compiled from three renowned fish datasets:
  -  **Deepfish** [^1]: This dataset comprises footage from 20 distinct underwater habitats, offering a diverse representation of underwater environments. This dataset was not originally intended for object detection tasks, which means it did not include bounding box annotations for fish. In the context of this project, the same dataset used by A.A. Muksit et al. [^4] was utilized. To adapt the DeepFish dataset for fish detection, A.A. Muksit et al. manually curated a subset of images, selecting those that exhibited different types of fish movement and posture across various habitats. From this selection process, they identified 4,505 positive images and then meticulously annotated a total of 15,463 ground truth bounding boxes within these images.
  -  **Fish4knowledge from Roboflow** [^2]: This dataset is derived from the larger Fish4Knowledge dataset and includes 1,879 frames that have been pre-processed and annotated specifically for use with the YOLOv8   format.
  -  **Ozfish** [^3]: This dataset was developed as part of the Australian Research Data Commons Data Discoveries program to support research in machine learning for automated fish detection from video footage. It contains approximately 43,000 bounding box annotations across nearly 1,800 frames, making it a substantial resource for object detection tasks. What sets the OzFish dataset apart is its diversity; each image includes multiple species and various shapes of fish, with an average of 25 fish per frame. Some frames feature between 80 and 120 fish, presenting a significant challenge due to the high density of fish objects, many of which are small and difficult to detect. In contrast, the DeepFish dataset typically has 3 to 4 fish per frame, with a maximum of about 14 in some cases. The prevalence of numerous tiny fish in OzFish makes it particularly challenging for detection algorithms, especially when it comes to accurately identifying and differentiating among the small, densely packed objects.

For more details, see the following <a href="https://github.com/Andyvince01/Underwater_Fish_Detection/tree/main/data#-fishscale-dataset"> README.md </a>. The proposed final model demonstrates superior performance compared to the baseline models. However, when tested on a completely different dataset, it became evident that the model's performance was significantly lower. Notably, the *FishScale dataset* is characterized by high-quality images, with a limited number of samples captured using low-resolution cameras. To address this issue and enhance the robustness of the system, an **additional private dataset** provided by my Department (DIEM, UNISA) was utilized.

## 🧮 Results
This section delves into the performance evaluation of various models trained on the FishScale dataset. The comparative analysis encompasses multiple models, all of which were assessed using a consistent test set—identical to the one employed by A.A. Muskit et al. in their research[^4]. To ensure a fair evaluation, all models were subjected to the same primary validation thresholds: **`conf_tresh = 0.15`** and **`nms_tresh = 0.6`**. These values were chosen to ensure a good trade-off between precision and recall.

### A.A. Muskit et al. [^4]
This subsection highlights the results achieved by the models developed by A.A. Muskit et al. trained on the combined Deepfish + Ozfish dataset. It is noteworthy that the results reported here diverge from those documented in their paper. The primary reason for this discrepancy lies in the application of different threshold settings: specifically, `conf_tresh = 0.25` and `nms_tresh = 0.45` were employed in their original work. This variation in threshold values significantly influences model performance metrics, such as precision, recall, and F1-score. As a result, the models from Muskit et al. may exhibit different detection capabilities than what is observed in this analysis, which utilizes stricter thresholds to refine the results.

| Model            | Precision ↑    | Recall ↑       | F1-Score ↑      | mAP(50) ↑       | mAP(50-95) ↑ | Parameters ↓ | Gflops ↓ |
|------------------|----------------|----------------|-----------------|-----------------|--------------|--------------|----------|
| YOLOv3           | 0.67           | <ins>0.72<ins> | 0.690           | 0.739           | ❌           | 61,576,342   | 139.496  |
| YOLO-Fish-1      | <ins>0.70<ins> | 0.71           | <ins>0.705<ins> | <ins>0.745<ins> | ❌           | 61,559,958   | 173.535  |
| YOLO-Fish-2      | **0.74**       | 0.69           | **0.714**       | 0.736           | ❌           | 62,610,582   | 174.343  |
| YOLOv4           | 0.59           | **0.79**       | 0.675           | **0.787**       | ❌           | 64,003,990   | 127.232  |

### Mine from scratch
The following table summarizes the results from models I developed from scratch, specifically tuned for fish detection, incorporating adaptations such as $P_2$ layer, `SPD-Conv` and `CBAM` modules to enhance performance. All these models were trained on the _FishScale Dataset_ + _Private Dataset_.

| Model                | Precision ↑      | Recall ↑       | F1-Score ↑      | mAP(50) ↑       | mAP(50-95) ↑   | Parameters ↓ | Gflops ↓ |
|----------------------|------------------|----------------|-----------------|-----------------|----------------|--------------|----------|
| YOLOv8s              | 0.85,6           | 0.706          | 0.774           | 0.822           | 0.51           | 11,125,971   | 28.40    |
| YOLOv8s-P2           | 0.85             | <ins>0.72<ins> | 0.779           | 0.829           | 0.519          | 10,626,708   | 36.60    |
| YOLOv8s-p2-SPD       | **0.867**        | 0.717          | **0.785**       | <ins>0.831<ins> | <ins>0.52<ins> | 12,187,284   | 55.70    |
| YOLOv8s-p2-CBAM      | 0.844            | 0.719          | 0.778           | 0.83            | 0.512          | 15,808,192   | 50.90    |
| **YOLOv8s-FishScale**| <ins>0.854</ins> | **0.725**      | <ins>0.784<ins> | **0.833**       | **0.529**      | 17,358,240   | 70.00    |

Although YOLOv8s-FishScale’s precision is slightly lower than YOLOv8s-p2-SPD's (85.4% vs. 86.7%), its higher recall and comparable F1-Score highlight a well-rounded performance profile. This makes YOLOv8s-FishScale suitable when the goal is a balance between minimizing false positives and capturing as many true instances as possible. However, YOLOv8s-p2-SPD remains competitive with slightly better precision and lower computational requirements.

### Merging YOLOv8s with YOLOv8s-FishScale
Fine-tuning YOLOv8s-FishScale using the weights from YOLOv8s proved to be ineffective due to the addition of new layers in FishScale, which prevented exact weight matching as required by YOLO. Consequently, to enhance performance, a weight-merging approach was adopted, structured as follows:
- For layers shared between YOLOv8s and YOLOv8s-FishScale, priority was given to the YOLOv8s weights.
- For the new layers introduced in YOLOv8s-FishScale, the model retained FishScale’s weights.
  
Fine-tuning the model with these merged weights led to an increase in performance across all metrics, as shown in the table below, demonstrating the effectiveness of the weight-merging approach:

| Model                   | Precision ↑ | Recall ↑ | F1-Score ↑ | mAP(50) ↑ | mAP(50-95) ↑ | Parameters ↓ | Gflops ↓ |
|-------------------------|-------------|----------|------------|-----------|--------------|--------------|----------|
| YOLOv8s-Fishscale †     | 0.853       | 0.736    | 0.79       | 0.839     | 0.537        | 17,358,240   | 70.00    |
| **YOLOv8s-Fishscale ☨** | **0.861**   |**0.738** | **0.795**  | **0.845**  | **0.542**   | 17,358,240   | 70.00    |

The difference between the two models lies in additional data augmentation techniques employed to further enhance performance.

### Integrating FUnIEGAN within YOLOv8s-FishScale backbone
Islam et al. [^5] proposed an innovative, yet straightforward, conditional GAN-based model designed to enhance underwater images. This models centers on a generator network that learns to map the distorted image $X$ to an enhanced output $Y$ trough a dynamic, adversarial relationship discriminator network.

The following models were trained for a reduced number of epochs (50 instead of 100) using the YOLOv8s-FishScale ☨ weights. As shown in the table, the **FUnIEGAN + YOLOv8s-Fishscale ☨** model, which was trained by freezing all encoder layers of FUnIEGAN except the last one and fine-tuning the remaining layers, achieved a slight improvement in F1-Score. All other performance metrics remained comparable to those of YOLOv8s-Fishscale ☨:

| Model                   | Precision ↑ | Recall ↑ | F1-Score ↑ | mAP(50) ↑ | mAP(50-95) ↑ | Parameters ↓ | Gflops ↓ |
|-------------------------|-------------|----------|------------|-----------|--------------|--------------|----------|
| FUnIEGAN (freezed) + ☨  | 0.857       | 0.695    | 0.767       | 0.816     | 0.519       | 24,388,355  | 198.40    |
| **FUnIEGAN + ☨**        | **0.867**   |**0.736** | **0.796**  | **0.843**  | **0.541**   | 24,388,355  | 198.40    |


[^1]: Saleh, Alzayat, Laradji, Issam H., Konovalov, Dmitry A., Bradley, Michael, Vazquez, David, Sheaves, Marcus, 2020. A realistic fish-habitat dataset to evaluate algorithms for underwater visual analysis. Sci. Rep. 10 (1), 1–10. [doi:10.53654/tangible.v5i1.110](https://doi.org/10.53654/tangible.v5i1.110).

[^2]: Fish4Knowledge Dataset. g18L5754. Fish4Knowledge Dataset. Open Source Dataset. Roboflow Universe, October 2023. Available at: https://universe.roboflow.com/g18l5754/fish4knowledge-dataset. 

[^3]: Australian Institute Of Marine Science, 2020. Ozfish dataset - machine learning dataset for baited remote underwater video stations.

[^4]: A. A. Muksit, F. Hasan, M. F. Hasan Bhuiyan Emon, M. R. Haque, A. R. Anwary, and S. Shatabda, “Yolo-fish: A robust fish detection model to detect fish in realistic underwater environment,” Ecological Informatics, vol. 72, p. 101847, 2022. [doi:10.1016/j.ecoinf.2022.101847](https://doi.org/10.1016/j.ecoinf.2022.101847).

[^5]: M. J. Islam, Y. Xia and J. Sattar, "Fast Underwater Image Enhancement for Improved Visual Perception," in IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 3227-3234, April 2020, [doi:10.1109/LRA.2020.2974710](https://doi.org/10.1109/LRA.2020.2974710).
