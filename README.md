# 🐠 YOLO-FishScale - Underwater Fish Detection (🏗️ Implementation in progres...)

This thesis aims to design a real-time system capable of processing the images acquire in real-time, detecting and tracking all fish present in the scene, while also identifying any behavioral patterns.

This study proposes a custom fish detection method based on the YOLOv8 algorithm, named **`YOLO-FishScale`**. Although YOLOv8 represents the current state-of-the-art (SOTA) in object detection, it is not without limitations. Given that fish detection is a multi-scale problem—where fish can be located at varying distances from the camera and can have different sizes—it is essential to effectively manage the presence of small fish. The original YOLOv8 model employs a backbone network that down-samples the image through five stages, resulting in five feature layers ($P_1$, $P_2$, $P_3$, $P_4$, and $P_5$). Here, each $P_i$ layer represents a resolution of $1/2^i$ of the original image. To address the challenge of detecting small fish more effectively, YOLO-FishScale proposes the addition of a new detection head to the YOLOv8 architecture. This additional detection head utilizes features from the $P_2$ _layer_, which is specifically designed to enhance micro-target detection capabilities. Small objects are particularly challenging to detect due to their low resolution, which provides limited information about the content needed to learn patterns. In YOLOv8, the feature extraction module Conv, a stride convolutional layer, rapidly degrades its detection performance when dealing with low-resolution images or small objects. To address this issue, this module is replaced with a new CNN building block, known as _SPD-Conv_ (space-to-depth), which has been shown to enhance performance in these complex tasks. Since small targets are small in size and have few and inconspicuous features. It is possible to enhance the feature fusion network by incorporating _CBAM_ (Convolutional Block Attention Mechanism) modules, which amplify global interactions and directly improve the feature fusion in the neck.

Underwater imaging presents significant challenges, primarily due to the absorption and scattering of light. These factors greatly reduce visibility, often limiting it to just a few meters. The unique optical properties of underwater environments cause light to behave differently than in air, resulting in images with diminished contrast and compromised color fidelity. This degradation makes it difficult to discern fine details, which is particularly problematic for tasks such as fish detection. To overcome these challenges, this thesis proposes the utilization of an existing model for underwater image enhancement—**`FUnIE-GAN`** (Fast Underwater Image Enhancement using Generative Adversarial Networks). FUnIE-GAN is a conditional GAN-based model specifically designed to enhance underwater images by addressing issues related to color, contrast, and clarity. The model achieves this by formulating a perceptual loss function that evaluates image quality based on several key aspects: global color, content, local texture, and style information. The model formulates a perceptual loss function by evaluating image quality based on its global color, content, local texture, and style information. In the proposed workflow, the acquired underwater images will first be preprocessed using FUnIE-GAN. Once enhanced, these images will then be fed into the custom YOLO-FishScale detection model, enabling more accurate fish detection and classification even in challenging underwater conditions. Aiming to produce better images for fish detection, an experiment will be made to fine-tune FUnIE-GAN using the loss from the YOLO-FishScale model. This approach seeks to enhance not only the visual quality of the images but also their effectiveness for detection tasks. However, this integration could introduce several challenges, including the need for architectural modifications to enable the gradient from YOLO-FishScale to backpropagate through the GAN.

Ultimately, a tracking algorithm—**`ByteTrack`**—will be integrated into the system to ensure continuous monitoring and analysis of the detected fish within the underwater scene. After YOLO-FishScale detects and classifies fish in individual frames, ByteTrack will maintain each fish's identity and location across consecutive frames, even in challenging conditions such as occlusions or abrupt movements. By integrating ByteTrack, the system will not only track individual fish but also facilitate crowd counting by aggregating the number of tracked individuals over time. This will enable a comprehensive understanding of fish populations and their interactions within the environment, providing valuable data on behavior, density, and movement patterns.
