# 🐠 YOLO-FishScale - Underwater Fish Detection (🏗️ Ongoing...)

Freshwater ecosystems are facing significant challenges due to the extinction of endemic species, often caused by the invasion of aggressive species in degraded habitats. Overfishing and unsustainable practices are further threatening fish populations, destabilizing aquatic environments. This crisis is rooted in the exponential growth of the human population, which intensifies environmental degradation. To address declining wild fish stocks, aquaculture has emerged as a vital solution, not only for food production but also for environmental conservation by restoring natural habitats and replenishing wild populations. In this context, deep learning techniques are revolutionizing aquaculture by enabling precise monitoring and management of aquatic environments. The ability to process and analyze large volumes of visual data in real-time helps in accurately detecting, tracking, and understanding fish behavior, which is crucial for both optimizing aquaculture practices and preserving natural ecosystems.

This thesis presents a real-time system for detecting and tracking fish in underwater environments, utilizing a custom fish detector called **`YOLO-FishScale`** based on the `YOLOv8` algorithm. This detector addresses the challenge of detecting small fish, which vary in size and distance within frames. It enhances YOLOv8 by adding a new detection head that uses features from the $P_2$ layer and replaces the Conv module with `SPD-Conv`, improving performance on small and low-resolution targets. Additionally, `CBAM` (Convolutional Block Attention Mechanism) modules are integrated to enhance feature fusion, resulting in more accurate fish detection and tracking.

![Screenshot_15-10-2024_17260_](https://github.com/user-attachments/assets/14a09688-8c0f-4486-a6e7-184c31093ff8)


>[!NOTE]
>The original YOLOv8 model employs a backbone network that down-samples the image through five stages, resulting in five feature layers ($P_1$, $P_2$, $P_3$, $P_4$, and $P_5$). Here, each $P_i$ layer represents a resolution of $1/2^i$ of the original image. To address the challenge of detecting small fish more effectively, `YOLO-FishScale` proposes the addition of a new detection head to the `YOLOv8` architecture. This additional detection head utilizes features from the $P_2$ _layer_, which is specifically designed to enhance micro-target detection capabilities. Small objects are particularly challenging to detect due to their low resolution, which provides limited information about the content needed to learn patterns.

## 🎣 FishScale Dataset
To train the `YOLO-FishScale` model,, a new comprehensive dataset was constructed. The **Fishscale Dataset** is an extensive collection of images and labels, compiled from three renowned fish datasets:
  -  **Deepfish** [^1]: This dataset comprises footage from 20 distinct underwater habitats, offering a diverse representation of underwater environments. This dataset was not originally intended for object detection tasks, which means it did not include bounding box annotations for fish. In the context of this project, the same dataset used by A.A. Muksit et al. [^4] was utilized. To adapt the DeepFish dataset for fish detection, A.A. Muksit et al. manually curated a subset of images, selecting those that exhibited different types of fish movement and posture across various habitats. From this selection process, they identified 4,505 positive images and then meticulously annotated a total of 15,463 ground truth bounding boxes within these images.
  -  **Fish4knowledge from Roboflow** [^2]: This dataset is derived from the larger Fish4Knowledge dataset and includes 1,879 frames that have been pre-processed and annotated specifically for use with the YOLOv8   format.
  -  **Ozfish** [^3]: This dataset was developed as part of the Australian Research Data Commons Data Discoveries program to support research in machine learning for automated fish detection from video footage. It contains approximately 43,000 bounding box annotations across nearly 1,800 frames, making it a substantial resource for object detection tasks. What sets the OzFish dataset apart is its diversity; each image includes multiple species and various shapes of fish, with an average of 25 fish per frame. Some frames feature between 80 and 120 fish, presenting a significant challenge due to the high density of fish objects, many of which are small and difficult to detect. In contrast, the DeepFish dataset typically has 3 to 4 fish per frame, with a maximum of about 14 in some cases. The prevalence of numerous tiny fish in OzFish makes it particularly challenging for detection algorithms, especially when it comes to accurately identifying and differentiating among the small, densely packed objects.

For more details, see the following <a href="https://github.com/Andyvince01/Underwater_Fish_Detection/tree/main/data#-fishscale-dataset"> README.md </a>.

## :abacus: Results
This section delves into the performance evaluation of various models trained on the FishScale dataset. The comparative analysis encompasses multiple models, all of which were assessed using a consistent test set—identical to the one employed by A.A. Muskit et al. in their research[^4]. To ensure a fair evaluation, all models were subjected to the same primary validation thresholds: **`conf_tresh = 0.15`** and **`nms_tresh = 0.6`**. These values were chosen to ensure a good trade-off between precision and recall.

### A.A. Muskit et al. [^4]
This subsection highlights the results achieved by the models developed by A.A. Muskit et al. trained on the combined Deepfish + Ozfish dataset. It is noteworthy that the results reported here diverge from those documented in their paper. The primary reason for this discrepancy lies in the application of different threshold settings: specifically, `conf_tresh = 0.25` and `nms_tresh = 0.45` were employed in their original work. This variation in threshold values significantly influences model performance metrics, such as precision, recall, and F1-score. As a result, the models from Muskit et al. may exhibit different detection capabilities than what is observed in this analysis, which utilizes stricter thresholds to refine the results.

| Model            | Precision (%) ↑ | Recall (%) ↑ | F1-Score (%) ↑ | AP(50)% ↑ | AP(50-95)% ↑ | Parameters ↓ | Gflops ↓ |
|------------------|-----------------|--------------|----------------|-----------|--------------|--------------|----------|
| YOLOv3           | 67.00           | 72.00        | 69.41          | 73.94     | ❌           | 61,576,342   | 139.496  |
| YOLO-Fish-1      | 70.00           | 71.00        | 70.50          | 74.50     | ❌           | 61,559,958   | 173.535  |
| YOLO-Fish-2      | **74.00**       | 69.00        | **71.41**      | 73.64     | ❌           | 62,610,582   | 174.343  |
| YOLOv4           | 59.00           | **79.00**    | 67.55          | **78.71** | ❌           | 64,003,990   | 127.232  |

### Mine from scratch
The following table summarizes the results from models I developed from scratch, specifically tuned for fish detection, incorporating adaptations such as $P_2$ layer, `SPD-Conv` and `CBAM` modules to enhance performance.

| Model                | Precision (%) ↑ | Recall (%) ↑ | F1-Score (%) ↑ | AP(50)% ↑ | AP(50-95)% ↑ | Parameters ↓ | Gflops ↓ |
|----------------------|-----------------|--------------|----------------|-----------|--------------|--------------|----------|
| YOLOv8s              | 84.90           | 69.00        | 76.13          | 81.00     | 49.80        | 11,125,971   | 28.40    |
| YOLOv8s-P2           | **85.00**       | 70.08        | 76.82          | 82.22     | 50.80        | 10,626,708   | 36.60    |
| YOLOv8s-p2-SPD       | 84.20           | 71.50        | 77.33          | 82.50     | 51.00        | 12,187,284   | 55.70    |
| YOLOv8s-p2-CBAM      | 84.44           | 70.70        | 76.96          | 82.10     | 51.20        | 15,808,192   | 50.90    |
| **YOLOv8s-FishScale**| 84.70           | **72.00**    | **77.84**      | **83.00** | **52.00**    | 17,358,240   | 70.00    |

The results from my custom YOLO implementations indicate a significant leap in performance compared to the previous models. With a precision of 84.70%, **`YOLOv8s-FishScale`** excels in minimizing false positives, ensuring that a high percentage of detected objects are indeed fish. The recall rate of 72.00% suggests that it successfully identifies a significant proportion of the actual fish present in the dataset, albeit with a slightly lower recall compared to its precision. This balance between precision and recall leads to a commendable F1-Score of 77.84%, indicating a well-rounded model performance that can be reliably deployed in real-world scenarios. The AP(50)% score of 83.00% illustrates the model’s effectiveness at a 50% IoU threshold, showcasing its reliability in detecting fish with acceptable overlaps. The AP(50-95)% score of 52.00% further confirms the model's robustness across varying levels of detection overlap, indicating that it maintains strong performance even under stricter evaluation conditions. Despite having a larger number of parameters (17,358,240) and higher Gflops (70.00), which generally implies increased computational requirements, the trade-off in terms of performance is worthwhile. 

[^1]: Saleh, Alzayat, Laradji, Issam H., Konovalov, Dmitry A., Bradley, Michael, Vazquez, David, Sheaves, Marcus, 2020. A realistic fish-habitat dataset to evaluate algorithms for underwater visual analysis. Sci. Rep. 10 (1), 1–10. [doi:10.53654/tangible.v5i1.110](https://doi.org/10.53654/tangible.v5i1.110).

[^2]: Fish4Knowledge Dataset. g18L5754. Fish4Knowledge Dataset. Open Source Dataset. Roboflow Universe, October 2023. Available at: https://universe.roboflow.com/g18l5754/fish4knowledge-dataset. 

[^3]: Australian Institute Of Marine Science, 2020. Ozfish dataset - machine learning dataset for baited remote underwater video stations.

[^4]: A. A. Muksit, F. Hasan, M. F. Hasan Bhuiyan Emon, M. R. Haque, A. R. Anwary, and S. Shatabda, “Yolo-fish: A robust fish detection model to detect fish in realistic underwater environment,” Ecological Informatics, vol. 72, p. 101847, 2022. [doi:10.1016/j.ecoinf.2022.101847](https://doi.org/10.1016/j.ecoinf.2022.101847).
