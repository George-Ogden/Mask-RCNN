# Mask-RCNN Implementation
This is a PyTorch implementation of Mask-RCNN. It uses the webcam or an input video and displays the output or saves the video. See [usage](#usage) for more information.
View the paper at [arxiv.org/abs/1703.06870v1](https://arxiv.org/abs/1703.06870v1) or the PyTorch model source code [pytorch.org/vision/stable/_modules/torchvision/models/detection/mask_rcnn.html](https://pytorch.org/vision/stable/_modules/torchvision/models/detection/mask_rcnn.html).
## Setup
### pip
`pip install -r requirements.txt`
### conda
`conda env create -f env.yaml`
## Usage
```
usage: main.py [-h] [--video VIDEO] [--grey-background] [--classes CLASSES [CLASSES ...]] [--output-path OUTPUT_PATH] [--no-save] [--detection-threshold DETECTION_THRESHOLD] [--mask-threshold MASK_THRESHOLD]
               [--max-detections MAX_DETECTIONS] [--hide-video | --display-title DISPLAY_TITLE] [--hide-boxes] [--hide-masks] [--hide-labels] [--mask-opacity MASK_OPACITY] [--output-fps OUTPUT_FPS]
               [--show-fps] [--text-thickness TEXT_THICKNESS] [--box-thickness BOX_THICKNESS]

Mask-RCNN (segmentation model) implementation in PyTorch

optional arguments:
  -h, --help            show this help message and exit
  --video VIDEO         input video feed (use 0 for webcam)
  --grey-background     make the background monochromatic
  --classes CLASSES [CLASSES ...]
                        limit to certain classes
  --output-path OUTPUT_PATH
                        video save location
  --no-save             do not save output video
  --detection-threshold DETECTION_THRESHOLD
                        confidence threshold for detection (0-1)
  --mask-threshold MASK_THRESHOLD
                        confidence threshold for segmentation mask (0-1)
  --max-detections MAX_DETECTIONS
                        maximum concurrent detections (leave 0 for unlimited)
  --hide-video          do not show output video
  --display-title DISPLAY_TITLE
                        window title
  --hide-boxes          do not show bounding boxes
  --hide-masks          do not show segmentation masks
  --hide-labels         do not show labels
  --mask-opacity MASK_OPACITY
                        opacity of segmentation masks
  --output-fps OUTPUT_FPS
                        output fps for video (for webcam speed only)
  --show-fps            display processing speed (fps)
  --text-thickness TEXT_THICKNESS
                        thickness of label text
  --box-thickness BOX_THICKNESS
                        thickness of boxes
```