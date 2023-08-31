# Vehicles and plates detector and tracking with string plate recognition

This repository contain the project for Vision and Perception exam of Master in Artificial in Intelligence and Robotics from Univeristy La Sapienza of Rome.

The string plate recognition part of this project has been developed and uploaded by [Luigi Gallo](https://github.com/luigi-ga), co-worker of this project, on the following repository https://github.com/luigi-ga/ALPRNet.git.

![Alt Text](media/gif_demo.gif)

## Description
The project implements the following features:
- object detector for vehicles and plates based on [FasterRCNN_ResNet50](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html) of PyTorch
- string plate recognition based on ALPRNet (imported as sub-module from https://github.com/luigi-ga/ALPRNet.git)
- tracking of vehicles based on string plate recognition
- velocity estimation based on domain knowledge

From this [link](https://drive.google.com/file/d/1z-f7ND6Y4OfmDalFB_t9bMxFPyOE21D3/view?usp=share_link) you can download the full demo video.

## Directory tree
```sh
  -- VehiclesAndPlates-Detector 
     |--- src
          |--- ALPRNet (sub-module)
          |--- modules
          |    |--- dataset.py
          |    |--- detect_plate_string.py
          |    |--- inference.py
          |    |--- model.py
          |    |--- real_time_object_detector.py
          |    |--- training.py
          |--- utils
          |    |--- frames_to_video.py
          |--- notebook.ipynb
          |--- test.py
```

## Installation 
1. Clone this repository:
```sh
git clone https://github.com/ValerioSpagnoli/VehiclesAndPlates-Detector.git
cd VehiclesAndPlates-Detector
```
2. Install the required dependencies using `pip`:
```sh
pip install -r requirements.txt
```

## Usage
Download the weights following the instructions in [Download weights](#download-weights).

From this [link](https://drive.google.com/file/d/1yx1Ou7iClEo5t-Ki9wWFVcgR7iKx_UbN/view?usp=share_link) download the demo video, and put it into a folder named ```video_test``` in the main directory of this repository. The directory three must be:
```sh
  -- VehiclesAndPlates-Detector
     |--- video_test
     |    |--- video_test.mp4  
     |--- src
          |--- ...
```
If you want use a different video open ```test.py``` and change the parameter ```video_path``` of the function ```real_time_object_detector``` with the path of your video.
Then, launch the following command from the main directory of this repository:
```sh
python src/test.py
```

## Download weights
This project use three networks:
- Faster-RCNN-ResNet50 for object detection
- EDSR for super resolution
- ALPRNet for string plate recognition
  
From the this [link](https://drive.google.com/drive/folders/1GNwxJwKyAZybAP71T0N0wo2yH6GAgzkP?usp=share_link) you can download three folders with all weights needed. Please create a folder named ```weights``` in the main directory of this repository and put all three folders downloaded into it. The directory tree must be:

```sh
  -- VehiclesAndPlates-Detector
     |--- weights
     |    |--- alpr_weights
     |    |--- detector_weights
     |    |--- edsr_weights
     |--- src
          |--- ...
```