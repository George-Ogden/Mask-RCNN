import os
import cv2
import time
import torch
import random

from torchvision.models.detection import maskrcnn_resnet50_fpn as maskrcnn
import numpy as np

if not os.path.exists("output"):
    os.mkdir("output")

model = maskrcnn(pretrained=True).eval()
classes = ["BG","person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant",None,"stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",None,"backpack","umbrella",None,None,"handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",None,"wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",None,"dining table",None,None,"toilet",None,"tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",None,"book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
colours = [[random.randint(0,255) for c in range(3)] for _ in range(len(classes))]

VIDEO = "bolt-short.mp4"
OUTPUT_PATH = "output/video.mp4"
OUTPUT_FPS = 10
DETECTION_THRESHOLD = 0.7
MASK_THRESHHOLD = 0.5
MAX_DETECTIONS = 2
BOX_THICKNESS = 3
TEXT_THICKNESS = 2
MASK_OPACITY = 0.4
DISPLAY_TITLE = "Mask-RCNN"

cap = cv2.VideoCapture(VIDEO)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), OUTPUT_FPS, (w, h))

while True:
    t0 = time.time()
    ret, image = cap.read()
    if not ret:
        break
    output = model(torch.tensor(np.expand_dims(image,axis=0)).permute(0,3,1,2) / 255)[0]
    for i, (box, label, score, mask) in enumerate(zip(*output.values())):
        if score < DETECTION_THRESHOLD or i == MAX_DETECTIONS:
            break
        image = cv2.rectangle(image,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),colours[label], BOX_THICKNESS)
        image = cv2.putText(image, classes[label], (int(box[0]),int(box[1]) - 3),0, 0.8, colours[label], TEXT_THICKNESS)
        image[mask[0] > MASK_THRESHHOLD] = image[mask[0] > MASK_THRESHHOLD] * (1 - MASK_OPACITY) + MASK_OPACITY * np.array(colours[label])
    cv2.imshow(DISPLAY_TITLE,image)
    writer.write(image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    print(f"FPS: {1/(time.time()-t0):.2f}"+" "*5,end="\r")
