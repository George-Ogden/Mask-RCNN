import os
import cv2
import time
import torch
import random
import argparse
from glob import glob

from torchvision.models.detection import maskrcnn_resnet50_fpn as maskrcnn
import numpy as np

# classes and randomly generated colours
classes = ["BG","person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant",None,"stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",None,"backpack","umbrella",None,None,"handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",None,"wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",None,"dining table",None,None,"toilet",None,"tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",None,"book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
colours = [[random.randint(0,255) for c in range(3)] for _ in range(len(classes))]

# default arguments
parser = argparse.ArgumentParser(description="Mask-RCNN (segmentation model) implementation in PyTorch")
output_group = parser.add_mutually_exclusive_group()
boxes_group = parser.add_mutually_exclusive_group()
masks_group = parser.add_mutually_exclusive_group()
labels_group = parser.add_mutually_exclusive_group()
parser.add_argument("--grey-background","-g",action="store_true",help="make the background monochromatic")
parser.add_argument("--classes","-c",nargs="+",default=["all"],help="limit to certain classes (all or see classes.txt)")
parser.add_argument("--detection-threshold",default=0.7,type=float,help="confidence threshold for detection (0-1)")
parser.add_argument("--mask-threshold",default=0.5,type=float,help="confidence threshold for segmentation mask (0-1)")
parser.add_argument("--max-detections",default=0,type=int,help="maximum concurrent detections (leave 0 for unlimited)")
output_group.add_argument("--hide-output",action="store_true",help="do not show output")
output_group.add_argument("--display-title",default="Mask-RCNN",help="window title")
boxes_group.add_argument("--hide-boxes",action="store_true",help="do not show bounding boxes")
masks_group.add_argument("--hide-masks",action="store_true",help="do not show segmentation masks")
labels_group.add_argument("--hide-labels",action="store_true",help="do not show labels")
masks_group.add_argument("--mask-opacity",default=0.4,type=float,help="opacity of segmentation masks")
parser.add_argument("--show-fps",action="store_true",help="display processing speed (fps)")
labels_group.add_argument("--text-thickness",default=2,type=int,help="thickness of label text")
boxes_group.add_argument("--box-thickness",default=3,type=int,help="thickness of boxes")

# subparsers for different inputs
subparsers = parser.add_subparsers()

image = subparsers.add_parser("image")
output_group = image.add_mutually_exclusive_group()
image.add_argument("--input-image","--input","-i",default="example.png",required=True,help="input image")
output_group.add_argument("--save-path","--output","-o",default="output.png",help="output save location")
output_group.add_argument("--no-save",action="store_true",help="do not save output image")
image.set_defaults(action="image")

folder = subparsers.add_parser("folder")
output_group = folder.add_mutually_exclusive_group()
folder.add_argument("--input-folder","--input","-i",default=".",required=True,help="input folder")
output_group.add_argument("--output-folder","--output","-o",default="output/",help="output save location")
output_group.add_argument("--no-save",action="store_true",help="do not save output images")
folder.add_argument("--extensions","-e",nargs="+",default=["png", "jpeg", "jpg", "bmp", "tiff", "tif"],help="image file extensions")
folder.set_defaults(action="folder")

video = subparsers.add_parser("video")
output_group = video.add_mutually_exclusive_group()
video.add_argument("--input-video","--input","-i",default="example.mp4",required=True,help="input video")
output_group.add_argument("--save-path","--output","-o",default="output.mp4",help="output save location")
output_group.add_argument("--no-save",action="store_true",help="do not save output video")
video.set_defaults(action="video")

webcam = subparsers.add_parser("webcam")
output_group = webcam.add_mutually_exclusive_group()
webcam.add_argument("--source","--input","-i",type=int,default=0,help="webcam number")
output_group.add_argument("--save-path","--output","-o",default="output.mp4",help="output save location")
webcam.add_argument("--output-fps",default=1,type=int,help="output fps for video")
output_group.add_argument("--no-save",action="store_true",help="do not save output video")
webcam.set_defaults(action="webcam")

# parse args
args = parser.parse_args()
include_classes = classes[1:] if "all" in args.classes else args.classes
mode = args.action

# load model
model = maskrcnn(pretrained=True).eval()

def detect(image):
    # feed forward the image
    output = model(torch.tensor(np.expand_dims(image,axis=0)).permute(0,3,1,2) / 255)[0]
    if args.grey_background:
        # create a cover
        cover = np.zeros(image.shape,dtype=bool)
    for i, (box, label, score, mask) in enumerate(zip(*output.values())):
        # check if we need to keep detecting
        if score < args.detection_threshold or (i >= args.max_detections and args.max_detections != 0):
            break
        # ignore irrelevant classes
        if not classes[label] in include_classes:
            continue
        # draw box
        if not args.hide_boxes:
            image = cv2.rectangle(image,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),colours[label], args.box_thickness)
        # draw label
        if not args.hide_labels:
            image = cv2.putText(image, classes[label], (int(box[0]),int(box[1]) - 3),0, 0.8, colours[label], args.text_thickness)
        # draw mask
        if not args.hide_masks:
            image[mask[0] > args.mask_threshold] = image[mask[0] > args.mask_threshold] * (1 - args.mask_opacity) + args.mask_opacity * np.array(colours[label])
        # update the cover
        if args.grey_background:
            cover[mask[0] > args.mask_threshold] = 1
    # make the background grey
    if args.grey_background:
        image[~cover] = np.tile(np.expand_dims(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),axis=2),(1,1,3))[~cover]
    return image
    
if mode in ["video","webcam"]:
    # find the correct source
    if mode == "video":
        source = args.input_video
    else:
        source = args.source
    # create a video capture
    cap = cv2.VideoCapture(source)
    if cap is None or not cap.isOpened():
        raise RuntimeError(f"video (\"{source}\") is not a valid input")
    # create an output writer
    if not args.no_save:
        # get the width and height
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # get the fps
        if mode == "video":
            output_fps = int(cap.get(cv2.CAP_PROP_FPS))
        else:
            output_fps = args.output_fps
        # create the video writer
        directory = os.path.dirname(args.save_path)
        if directory:
            os.makedirs(directory,exist_ok=True)
        writer = cv2.VideoWriter(args.save_path, cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (w, h))
    mode = "video"
elif mode == "folder":
    files = []
    folder = args.input_folder
    # add a "/" to the end of the folder
    if not folder.endswith("/") and not folder.endswith("\\"):
        folder += "/"
    # create a list of files
    for extension in args.extensions:
        files += glob(f"{folder}**/*.{extension}",recursive=True)
    i = 0

t0 = time.time()
while True:
    # read the next frame
    if mode == "video":
        ret, image = cap.read()
        if not ret:
            break
    elif mode == "folder":
        # get the path and image
        path = files[i]
        image = cv2.imread(path)
        i += 1 # increment counter
    else:
        # read the image
        image = cv2.imread(args.input_image)
    # run the detection
    image = detect(image)
    # show the image
    if not args.hide_output:
        cv2.imshow(args.display_title,image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    # save output
    if not args.no_save:
        # write using video writer
        if mode == "video":
            writer.write(image)
        else:
            # create a save path
            if mode == "folder":
                save_path = os.path.join(args.output_folder,os.path.relpath(path,folder))
            else:
                save_path = args.save_path
            # save image
            directory = os.path.dirname(save_path)
            if directory:
                os.makedirs(directory,exist_ok=True)
            cv2.imwrite(save_path,image)
            if mode == "image" or i >= len(files):
                break
    # calculate and print fps
    if args.show_fps:
        print(f"FPS: {1/(time.time()-t0):.2f}"+" "*5,end="\r")
        t0 = time.time()