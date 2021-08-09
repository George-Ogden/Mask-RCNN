import os
import cv2
import time
import torch
import random
import argparse

from torchvision.models.detection import maskrcnn_resnet50_fpn as maskrcnn
import numpy as np

classes = ["BG","person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant",None,"stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",None,"backpack","umbrella",None,None,"handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",None,"wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",None,"dining table",None,None,"toilet",None,"tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",None,"book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
colours = [[random.randint(0,255) for c in range(3)] for _ in range(len(classes))]

parser = argparse.ArgumentParser(description="Mask-RCNN (segmentation model) implementation in PyTorch")
video_group = parser.add_mutually_exclusive_group()
boxes_group = parser.add_mutually_exclusive_group()
masks_group = parser.add_mutually_exclusive_group()
labels_group = parser.add_mutually_exclusive_group()
parser.add_argument("--video",default="0",help="input video feed (use 0 for webcam)")
parser.add_argument("--grey-background",action="store_true",help="make the background monochromatic")
parser.add_argument("--classes",nargs="+",default=["all"],help="limit to certain classes")
parser.add_argument("--output-path",default="output/maskrcnn.mp4",help="video save location")
parser.add_argument("--no-save",action="store_true",help="do not save output video")
parser.add_argument("--detection-threshold",default=0.7,type=float,help="confidence threshold for detection (0-1)")
parser.add_argument("--mask-threshold",default=0.5,type=float,help="confidence threshold for segmentation mask (0-1)")
parser.add_argument("--max-detections",default=0,type=int,help="maximum concurrent detections (leave 0 for unlimited)")
video_group.add_argument("--hide-video",action="store_true",help="do not show output video")
video_group.add_argument("--display-title",default="Mask-RCNN",help="window title")
boxes_group.add_argument("--hide-boxes",action="store_true",help="do not show bounding boxes")
masks_group.add_argument("--hide-masks",action="store_true",help="do not show segmentation masks")
labels_group.add_argument("--hide-labels",action="store_true",help="do not show labels")
masks_group.add_argument("--mask-opacity",default=0.4,type=float,help="opacity of segmentation masks")
parser.add_argument("--output-fps",default=10,type=int,help="output fps for video (for webcam speed only)")
parser.add_argument("--show-fps",action="store_true",help="display processing speed (fps)")
labels_group.add_argument("--text-thickness",default=2,type=int,help="thickness of label text")
boxes_group.add_argument("--box-thickness",default=3,type=int,help="thickness of boxes")

args = parser.parse_args()

try:
    VIDEO = int(args.video)
    OUTPUT_FPS = args.output_fps
except:
    VIDEO = args.video
    OUTPUT_FPS = 0

OUTPUT_PATH = args.output_path
DETECTION_THRESHOLD = args.detection_threshold
MASK_THRESHHOLD = args.mask_threshold
MAX_DETECTIONS = args.max_detections
BOX_THICKNESS = args.box_thickness
TEXT_THICKNESS = args.text_thickness
MASK_OPACITY = args.mask_opacity
DISPLAY_TITLE = args.display_title
HIDE_BOXES = args.hide_boxes
HIDE_MASKS = args.hide_masks
HIDE_LABELS = args.hide_labels
HIDE_VIDEO = args.hide_video
NO_SAVE = args.no_save
SHOW_FPS = args.show_fps
GREY_BACKGROUND = args.grey_background
INCLUDE_CLASSES = classes[1:] if "all" in args.classes else args.classes

cap = cv2.VideoCapture(VIDEO)
if cap is None or not cap.isOpened():
    raise RuntimeError(f"video (\"{VIDEO}\") is not a valid video")
if not NO_SAVE:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if OUTPUT_FPS == 0:
        OUTPUT_FPS = int(cap.get(cv2.CAP_PROP_FPS))
    writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), OUTPUT_FPS, (w, h))
model = maskrcnn(pretrained=True).eval()
directory = os.path.dirname(OUTPUT_PATH)
os.makedirs(directory,exist_ok=True)

t0 = time.time()
while True:
    ret, image = cap.read()
    if not ret:
        break
    output = model(torch.tensor(np.expand_dims(image,axis=0)).permute(0,3,1,2) / 255)[0]
    if GREY_BACKGROUND:
        cover = np.zeros(image.shape,dtype=bool)
    for i, (box, label, score, mask) in enumerate(zip(*output.values())):
        if score < DETECTION_THRESHOLD or (i >= MAX_DETECTIONS and MAX_DETECTIONS != 0):
            break
        if not classes[label] in INCLUDE_CLASSES:
            continue
        if not HIDE_BOXES:
            image = cv2.rectangle(image,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),colours[label], BOX_THICKNESS)
        if not HIDE_LABELS:
            image = cv2.putText(image, classes[label], (int(box[0]),int(box[1]) - 3),0, 0.8, colours[label], TEXT_THICKNESS)
        if not HIDE_MASKS:
            image[mask[0] > MASK_THRESHHOLD] = image[mask[0] > MASK_THRESHHOLD] * (1 - MASK_OPACITY) + MASK_OPACITY * np.array(colours[label])
        if GREY_BACKGROUND:
            cover[mask[0] > MASK_THRESHHOLD] = 1
    if GREY_BACKGROUND:
        image[~cover] = np.tile(np.expand_dims(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),axis=2),(1,1,3))[~cover]
    if not HIDE_VIDEO:
        cv2.imshow(DISPLAY_TITLE,image)
    if not NO_SAVE:
        writer.write(image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    if SHOW_FPS:
        print(f"FPS: {1/(time.time()-t0):.2f}"+" "*5,end="\r")
        t0 = time.time()
