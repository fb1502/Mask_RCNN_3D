import cv2
import numpy as np
import os
import sys
from samples import coco
from mrcnn import utils
from mrcnn import model as modellib

class MRCNN3d():
    """
    Encapsulates the Mask RCNN 3d model functionality.
    """
    def __init__(self, input_file, output_file, bar_width, bar_distance, bar_color):
        self.bar_width = bar_width
        self.bar_distance = bar_distance
        self.bar_color = bar_color
        self.input_file = input_file
        self.output_file = output_file
        self.class_names = [
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
        self.ROOT_DIR = os.getcwd()
        self.MODEL_DIR = os.path.join(self.ROOT_DIR, "logs")
        self.COCO_MODEL_PATH = os.path.join(self.ROOT_DIR, "mask_rcnn_coco.h5")
        if not os.path.exists(self.COCO_MODEL_PATH):
            utils.download_trained_weights(self.COCO_MODEL_PATH)

        class InferenceConfig(coco.CocoConfig):
            GPU_COUNT = 1
            
            # Number of images to train with on each GPU. A 12GB GPU can typically
            # handle 2 images of 1024x1024px.
            # Adjust based on your GPU memory and image sizes. Use the highest
            # number that your GPU can handle for best performance.
            IMAGES_PER_GPU = 1

            # calculate batch_size for GPU usage
            def batch_size(self):
                return InferenceConfig.GPU_COUNT*InferenceConfig.IMAGES_PER_GPU

        # Change the config info            
        self.config = InferenceConfig()
        self.batch_size = self.config.batch_size()

        # COCO dataset object names
        self.model = modellib.MaskRCNN(mode="inference", model_dir=self.MODEL_DIR, config=self.config)
        self.model.load_weights(self.COCO_MODEL_PATH, by_name=True)

    # apply bar mask
    def apply_mask(self, image, mask):

        shape = self.video_shape
        bar_mask = np.full(shape, False, dtype=bool)

        position_1 = int(shape[1]*(.5-self.bar_distance/100))
        position_2 = int(shape[1]*(.5+self.bar_distance/100))
        bar_mask[:,position_1:position_1+ self.bar_width] = True
        bar_mask[:,position_2-self.bar_width:position_2] = True

        mask = mask | (~bar_mask)
        if self.bar_color == "white":
            background_color = np.full(shape, 255, dtype=int)
        else:
            background_color = np.full(shape, 0, dtype=int)

        image[:, :, 0] = np.where(
            mask == 0,
            background_color[:, :],
            image[:, :, 0]
        )
        image[:, :, 1] = np.where(
            mask == 0,
            background_color[:, :],
            image[:, :, 1]
        )
        image[:, :, 2] = np.where(
            mask == 0,
            background_color[:, :],
            image[:, :, 2]
        )
        return image

    # This function is used to show the object detection result in original image.
    def display_instances(self, image, boxes, masks, ids, names, scores):

        # n_instances saves the amount of all objects
        n_instances = boxes.shape[0]

        # print("there are n_instances in this picture:",n_instances)

        # if not n_instances:
        #     print('NO INSTANCES TO DISPLAY')
        # else:
        #     assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

        # initialize an empty mask
        mask = np.full(self.video_shape, False, dtype=bool)

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
        image = self.apply_mask(image, mask)
            
        return image

    def convert(self):
        capture = cv2.VideoCapture(self.input_file)

        # Recording Video
        fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(3))
        height = int(capture.get(4))
        self.video_shape = (height, width)
        fcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(self.output_file, fcc, fps, (width, height))
        count_frame = 0

        while True:
            print("*")
            count_frame += 1
            if count_frame % int(fps) == 0:
                print("\nProcessed: {0}%".format(round((count_frame) * 100 / total_frames)))

            ret, frame = capture.read()

            if not ret:
                break

            results = self.model.detect([frame], verbose=0)
            r = results[0]
            frame = self.display_instances(
                frame, r['rois'], r['masks'], r['class_ids'], self.class_names, r['scores']
            )
            # Recording Video
            out.write(frame)

        capture.release()

