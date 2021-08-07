import os
import cv2
import time
import torch
import random

import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn as maskrcnn

if not os.path.exists("output"):
    os.mkdir("output")

model = maskrcnn(pretrained=True).eval()
classes = ["BG","person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
classes = ["BG","person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant",None,"stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",None,"backpack","umbrella",None,None,"handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",None,"wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",None,"dining table",None,None,"toilet",None,"tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",None,"book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
colours = [[random.randint(0,255) for c in range(3)] for _ in range(len(classes))]

cap = cv2.VideoCapture(0)
cap.set(3, 810)
cap.set(4, 720)
file_path= "output/video"

file_path = "output/video"
i = 1
while os.path.exists(f"{file_path}-{i:02}.mp4"):
    i += 1
save_path = f"{file_path}-{i:02}.mp4"
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w * 2, h))

while True:
    t0 = time.time()
    ret, image = cap.read()
    if not ret:
        break
    output = model(torch.tensor(np.expand_dims(image,axis=0)).permute(0,3,1,2) / 255)[0]
    for box, label, score, mask in zip(*output.values()):
        if score < 0.7:
            break
        image = cv2.rectangle(image,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),colours[label], 3)
        image = cv2.putText(image, classes[label], (int(box[0]),int(box[1]) - 3),0, 0.8, colours[label],2)
        image[mask[0] > 0.5] = image[mask[0] > 0.5] * 0.75 + 0.25 * np.array(colours[label])
    cv2.imshow("Mask-RCNN",image)
    writer.write(image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    print(f"FPS: {1/(time.time()-t0):.2f}"+" "*5,end="\r")
