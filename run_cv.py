# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
import matplotlib.pyplot as plt
from Read_image import read_images_add
import datetime

def detect(image, net):
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)

    net.setInput(blob)
    layerOutputs = net.forward(ln)
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                            args["threshold"])

    return boxes, confidences, classIDs, idxs



def plot_results(pil_img, prob, boxes, classIDs, Name_model = 'DeTr',  Save_Images = True, image_name='test', inferene_time = None):
    if Save_Images :
        os.makedirs(Name_model,exist_ok=True)

    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    ######################
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():

            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], prob[i])
            print(text)
            #print(str(LABELS[classIDs[i]]) + ' : ' + str(confidences[i]))
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.putText(image, f"inference time : {inferene_time:0.2f}", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 1)
        cv2.putText(image, f"Number Objects : {len(idxs)}", (10, 25 +30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 1)
        # plt.axis('off')
        if Save_Images :
            addr_image = os.path.join(Name_model,image_name)
            cv2.imwrite(addr_image, image)
        else:
            cv2.imshow(f'image_name', image)
            cv2.waitKey(0)
        # plt.savefig(addr_image)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
#                 help="path to input image")
# ap.add_argument("-y", "--yolo", required=True,
#                 help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.15,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.7,
                help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
#labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
base_add = 'yolov3-tiny/'
labelsPath = base_add + 'coco.names'
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")
weightsPath = base_add + "yolov3.weights"
configPath = base_add + "yolov3.cfg"
# confidence = 0.3
# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

Time_inference = 0
allImageInferenceTime = 0
# Name_model = args["yolo"] + '_' +  str(args["confidence"])
Name_model = "yolo" + '_' +  str(args["confidence"])
Save_Images = True

images = read_images_add()
for addr in images[0]:
    image_name = os.path.split(addr)[1]
    image = cv2.imread(addr)
    (H, W) = image.shape[:2]
    start_time = time.time()
    boxes, scores, classes, idxs =  detect(image, net)
    Time_inference  = (time.time() - start_time)
    allImageInferenceTime += Time_inference
    plot_results(image, scores, boxes, classes, Name_model, Save_Images, image_name, Time_inference)


with open('inference_time.txt', 'a') as f:
    f.write(str(datetime.datetime.now()))
    f.writelines(f"------> Average inference time : {allImageInferenceTime/len(images[0])}   ,    ")
    f.writelines(f"all images's inference time : {allImageInferenceTime} \n")