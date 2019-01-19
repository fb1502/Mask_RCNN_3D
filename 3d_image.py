# USAGE
# python mask_rcnn.py 3d_image.py -i images/persons.jpe -o images/persons_3d.jpe
# python mask_rcnn.py 3d_image.py -i images/persons.jpe -o images/persons_3d.jpe -w 30 -d 20

# import the necessary packages
import cv2
import numpy as np
import os
import sys
import argparse
from samples import coco
from mrcnn import utils
from mrcnn import model as modellib

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True, help = "path to the input image file")
ap.add_argument("-o", "--output", required = True, help = "path to the output image file")
ap.add_argument("-w", "--width", help = "pixel width of the blank bar", default = 20)
ap.add_argument("-d", "--distance", type=int, choices=range(10,41), default = 15, help = "distance of the bar from the center, choice from int[10, 40]")

args = vars(ap.parse_args())
original_image = args["input"]
output_image = args["output"]
bar_width = int(args["width"])
bar_distance = int(args["distance"])

# read image
image = cv2.imread(original_image)
print("Read original image successfully! The original image shape is:")
print(image.shape)

# 2 gray image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("Change gray image successfully! The gray image shape is:")
print(gray_image.shape)

# Load the pre-trained model data
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Change the config infermation
class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    
    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 1
    
config = InferenceConfig()

# COCO dataset object names
model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)
model.load_weights(COCO_MODEL_PATH, by_name=True)
class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

def apply_mask(image, mask):

    shape = gray_image.shape
	bar_mask = np.full(shape, False, dtype=bool)

	position_1 = int(shape[1]*(.5-bar_distance/100))
	position_2 = int(shape[1]*(.5+bar_distance/100))
	bar_mask[:,position_1:position_1+bar_width] = True
	bar_mask[:,position_2-bar_width:position_2] = True

	mask = mask | (~bar_mask)
	background_white = np.full(shape, 255, dtype=int)

	image[:, :, 0] = np.where(
		mask == 0,
		background_white[:, :],
		image[:, :, 0]
	)
	image[:, :, 1] = np.where(
        mask == 0,
        background_white[:, :],
        image[:, :, 1]
    )
	image[:, :, 2] = np.where(
        mask == 0,
        background_white[:, :],
        image[:, :, 2]
    )
	return image

def display_instances(image, boxes, masks, ids, names, scores):

    # n_instances saves the amount of all objects
    n_instances = boxes.shape[0]
    print("there are n_instances in this picture:",n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    # initialize an empty mask
    mask = np.full(gray_image.shape, False, dtype=bool)

    for i in range(n_instances):

        if not np.any(boxes[i]):
            continue

        # use label to select person object from all the 80 classes in COCO dataset
        label = names[ids[i]]

        # merge masks with labels 'person'
        if label == 'person':
	        mask = mask | masks[:, :, i]
        else:
            continue

    # apply mask for the image
    image = apply_mask(image, mask)
        
    return image

results = model.detect([image], verbose=0)
r = results[0]
frame = display_instances(
    image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
)
cv2.imwrite(output_image, frame)
