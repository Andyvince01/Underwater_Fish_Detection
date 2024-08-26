# 🐠 OzFish Dataset

> [1] Australian Institute of Marine Science (AIMS), University of Western Australia (UWA) and Curtin University. (2019), OzFish Dataset - Machine learning dataset for Baited Remote Underwater Video Stations, https://doi.org/10.25845/5e28f062c5097

**OzFish** [1] is a collection of ~80k fish crops, ~45k bounding box annotations derived from Baited Remote Underwater Video Stations (BRUVS) and comprised of 70 families, 200 genera and 507 species of fish. This dataset is completely open and free to use for advancing machine learning for the classification of fish from underwater imagery.

## ⏹️ Bounding Box Annotations

![Bounding box annotations](https://open-AIMS.github.io/ozfish/bounding-box-annotations.png?raw=true "Bounding box annotations")

Bounding box annotations were generated on the Sagemaker Ground Truth Platform, using multiple observers and combining the results. Unlike the crops, frames and videos, these annotations are fish/no-fish only and have no species/genus/family labels. The images are available [here](https://data.pawsey.org.au/public/?path=/FDFML/labelled/frames) and metadata [here](https://data.pawsey.org.au/public/?path=/FDFML/labelled/manifests).

Bounding boxes have associated JSON metadata.

```json
{
    "source-ref":"E000501_R.MP4.31568.png",
    "20191014":{
        "annotations":[
            {"class_id":0,"width":139,"top":306,"height":84,"left":588.5},
            {"class_id":0,"width":229.5,"top":357,"height":331,"left":1151},
            {"class_id":0,"width":198.5,"top":745.5,"height":271, "left":823},
            {"class_id":0,"width":159.5,"top":806,"height":148.5,"left":0},
            {"class_id":0,"width":1014,"top":399.5,"height":395,"left":108.5}
        ],
        "image_size":[
            {"width":1920,"depth":3,"height":1080}
        ]},
        "20191014-metadata":{
            "class-map":{"0":"fish"},
            "human-annotated":"yes",
            "objects":[
                {"confidence":0.27},
                {"confidence":0.27},
                {"confidence":0.2},
                {"confidence":0.27},
                {"confidence":0.28}
            ],
            "creation-date":"2019-10-15T05:40:28.278830",
            "type":"groundtruth/object-detection"
        }
    }
```

However, to train a model using **`YOLOv8`**, the format for the annotations needs to be different from this JSON metadata format. It requires annotations in a text-based format, where each annotation file corresponds to an image and contains the bounding box information in a specific format. This text file contains *annotation lines*, where each line represents one bounding box in the image. The format of each line is:

```
<class_id> <x_center> <y_center> <width> <height>
```

where:

- **`class_id`**: The integer representing the class of the object.
- **`x_center`**: The normalized x-coordinate of the center of the bounding box (relative to the width of the image).
- **`y_center`**: The normalized y-coordinate of the center of the bounding box (relative to the height of the image).
- **`width`**: The normalized width of the bounding box (relative to the width of the image).
- **`height`**: The normalized height of the bounding box (relative to the height of the image).