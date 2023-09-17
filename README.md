# EyeRoad: Vehicles and plates detector and tracking with string plate recognition

This repository contain the project for Vision and Perception exam of Master in Artificial in Intelligence and Robotics from **University La Sapienza of Rome**.

The string plate recognition part of this project has been developed and uploaded by **[Luigi Gallo](https://github.com/luigi-ga)**, **co-worker of this project**, on the following repository https://github.com/luigi-ga/ALPRNet.git.

## Visuals

![Alt Text](media/demo.gif)

From this [link](https://drive.google.com/file/d/14FUnilJ6lGWAUMs6i-etV0Tw7AkkKlH2/view?usp=share_link) you can download the full demo video.

## Description

The project implements the following features:

- object detector of vehicles and plates based on [FasterRCNN_ResNet50](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html) of PyTorch:
  - By Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun, "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks". [[PDF](https://arxiv.org/pdf/1506.01497.pdf)]
- string plate recognition based on ALPRNet (imported as sub-module from https://github.com/luigi-ga/ALPRNet.git)
- tracking and counting of vehicles based on string plate recognition
- velocity estimation based on domain knowledge

The following **pseudocode** shows the inference stage of this project with reference to ``real_time_object_detector.py`` file.

The training stage is not explained here, but if you are interested check the following files: ``dataset.py``, ``model.py``, ``training.py`` and ``notebook.py``:

- in ``dataset.py`` there are classes and functions used to import dataset and create dataloaders
- in ``model.py`` there is the import of the pretrained model from PyTorch
- in ``training.py`` there is the training procedure

The file ``notebook.py`` was used as main to perform both training and inference.

```python
for frame in video:
  
  # use the FasterRCNNResNet50 model in inference mode to perform object detection of 
  # vehicles and plates using the frame image
  bounding_boxes_vehicles_and_plates = FasterRCNNResNet50(frame)

  # discard all bounding boxes of plates and apply non-maximum-suppresion and 
  # score-thresholding on bounding boxes of vehicles
  bounding_boxes_vehicles = decode_preditcion_vehicles(bounding_boxes_vehicles_and_plates)

  for bounding_box_vehicle in bounding_boxes_vehicles:

    # crop the image of vehicle using the bounding box
    cropped_vehicle = frame[bounding_box_vehicle]

    # use the FasterRCNNResNet50 model in inference mode to perform object detection of 
    # vehicles and plates using the cropped image
    bounding_boxes_vehicles_plates = FasterRCNNResNet50(cropped_vehicle)

    # discard all bounding boxes of vehicles (there should be no bb of vehicles at 
    # this stage) and extraxt the bounding box of the plate with the highest score, 
    # applying score-thresholding
    bounding_box_plate = decode_prediction_plates(bounding_boxes_vehicles_plates)

    # crop the image of plate using the bounding box
    cropped_plate = cropped_vehicle[bounding_box_plate]
  
    # apply super resolution model on cropped plate to obtain a better image
    sr_cropped_plate = EDSR(cropped_plate)

    # transfrom the sr_cropped_plate in a gray image
    gray_sr_cropped_plate = to_gray(sr_cropped_plate)

    # use ALPRNet model in inference mode to extract the string of the plate
    plate_string = ALPRNet(gray_sr_cropped_plate)

    # use the plate string to perform tracking on the vehicle with that plate
    idx = tracking(plate_string, bounding_box_vehicle)

    # detect the velocity of this vehicle
    velocity_detected = velocity_detector(idx, bounding_box_vehicle)
```

For the super resolution of the plate was used the EDSR model implemented in OpenCV:

- Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee, "Enhanced Deep Residual Networks for Single Image Super-Resolution," 2nd NTIRE: New Trends in Image Restoration and Enhancement workshop and challenge on image super-resolution in conjunction with CVPR 2017. [[PDF](https://arxiv.org/pdf/1707.02921.pdf)]
- [Official repository GitHub](https://github.com/sanghyun-son/EDSR-PyTorch)
- [OpenCV documentation](https://docs.opencv.org/4.x/d8/d11/classcv_1_1dnn__superres_1_1DnnSuperResImpl.html)

## Directory tree

```sh
EyeRoad
└── src
    ├── ALPRNet (sub-module)
    ├── modules
    │   ├── dataset.py
    │   ├── detect_plate_string.py
    │   ├── inference.py
    │   ├── model.py
    │   ├── real_time_object_detector.py
    │   └── training.py
    ├── utils
    │   └── frames_to_video.py 
    ├── notebook.ipynb
    └── test.py
```

## Installation

1. Clone this repository:

```sh
git clone --recurse-submodules https://github.com/ValerioSpagnoli/EyeRoad.git
cd EyeRoad
```

2. Install the required dependencies using `pip`:

```sh
pip install -r requirements.txt
```

## Usage

Download the weights following the instructions in [Download weights](#download-weights).

From this [link](https://drive.google.com/file/d/1yx1Ou7iClEo5t-Ki9wWFVcgR7iKx_UbN/view?usp=share_link) download the test video, and put it into a folder named ``video_test`` in the main directory of this repository. The directory three must be:

```sh
EyEroad
└── src
│   └── ...
└── video_test
    └── video_test.mp4
```

If you want use a different video open ``test.py`` and change the parameter ``video_path`` of the function ``real_time_object_detector`` with the path of your video.
Then, launch the following command from the main directory of this repository:

```sh
python src/test.py
```

## Download weights

This project use three networks:

- Faster-RCNN-ResNet50 for object detection
- EDSR for super resolution
- ALPRNet for string plate recognition

From this [link](https://drive.google.com/drive/folders/1GNwxJwKyAZybAP71T0N0wo2yH6GAgzkP?usp=share_link) you can download three folders with all weights needed. Please create a folder named ``weights`` in the main directory of this repository and put all three folders downloaded into it. The directory tree must be:

```sh
EyeRoad
└── src
│   └── ...
└── weights
    ├── alpr_weights
    ├── detector_weights
    └── edsr_weights
```
