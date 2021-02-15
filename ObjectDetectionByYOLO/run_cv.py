'''This is th first code of Object Detections. This work with Yolo3'''

# import the necessary packages
import argparse
import time
import os
import datetime
import matplotlib.pyplot as plt
import cv2
import numpy as np
from image_reader import read_images_add


def detect(read_image, net):
    """Detect Pics"""
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(read_image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(layer_names)
    boxes = []
    confidences = []
    class_ids = []
    # loop over each of the layer outputs
    for output in layer_outputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (center_x, center_y, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                box_x = int(center_x - (width / 2))
                box_y = int(center_y - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([box_x, box_y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                            args["threshold"])
    return boxes, confidences, class_ids, idxs
def plot_results(pil_img, prob, boxes, class_ids,\
                 name_model = 'DeTr',\
		 save_images = True,\
		 image_name='test',\
		 inferene_time = None):
    '''Plot Results'''
    if save_images :
        os.makedirs(name_model,exist_ok=True)
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    #output_pic = plt.gca()
    ######################
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            (box_x, box_y) = (boxes[i][0], boxes[i][1])
            (box_w, box_h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[class_ids[i]]]
            cv2.rectangle(read_image, (box_x, box_y), (box_x + box_w, box_y + box_h), color, 2)
            text = "{}: {:.4f}".format(LABELS[class_ids[i]], prob[i])
            print(text)
            #print(str(LABELS[class_ids[i]]) + ' : ' + str(confidences[i]))
            cv2.putText(read_image, text, (box_x, box_y - 5),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(read_image, f"inference time : {inferene_time:0.2f}",\
        (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 1)
        cv2.putText(read_image, f"Number Objects : {len(idxs)}",\
        (10, 25 +30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 1)
        # plt.axis('off')
        if save_images :
            addr_image = os.path.join(name_model,image_name)
            cv2.imwrite(addr_image, read_image)
        else:
            cv2.imshow(image_name, read_image)
            #cv2.imshow(f'image_name', read_image)
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
#LABELS_PATH = os.path.sep.join([args["yolo"], "coco.names"])
BASE_ADD = 'yolov3-tiny/'
LABELS_PATH = BASE_ADD + 'coco.names'
LABELS = open(LABELS_PATH).read().strip().split("\n")
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")
WEIGHTS_PATH = BASE_ADD + "yolov3.weights"
CONFIG_PATH = BASE_ADD + "yolov3.cfg"
# confidence = 0.3
# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
TIME_INFERENCE = 0
ALLIMAGEINFERENCETIME = 0
# name_model = args["yolo"] + '_' +  str(args["confidence"])
NAME_MODEL = "yolo" + '_' +  str(args["confidence"])
SAVE_IMAGES = True
images = read_images_add()

for addr in images:
    image_name = os.path.split(addr)[1]
    read_image = cv2.imread(addr)
    (H, W) = read_image.shape[:2]
    start_time = time.time()
    boxes, scores, classes, idxs =  detect(read_image, net)
    TIME_INFERENCE  = (time.time() - start_time)
    ALLIMAGEINFERENCETIME += TIME_INFERENCE
    plot_results(read_image, scores, boxes, classes,\
    NAME_MODEL, SAVE_IMAGES, image_name, TIME_INFERENCE)
with open('inference_time.txt', 'a') as f:
    f.write(str(datetime.datetime.now()))
    f.writelines(f"------> Average inference time : {ALLIMAGEINFERENCETIME/len(images[0])}   ,    ")
    f.writelines(f"all images's inference time : {ALLIMAGEINFERENCETIME} \n")