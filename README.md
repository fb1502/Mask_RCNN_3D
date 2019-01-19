# Mask-RCNN 3D

## Introduction

|**3d gif 1**|**3d gif 2**|**3d gif 3**|
| :--: | :--: | :--: | 
|![](images/fun3dgif_1.gif)|![](images/fun3dgif_2.gif)|![](images/fun3dgif_3.gif)|

As shown above, 


Here are some demo images:

## Usage

Before start, please download the mask_rcnn_coco.h5 file from the [released page](https://github.com/matterport/Mask_RCNN/releases) and put it in the main folder.

For creating 3D effect image

python mask_rcnn.py 3d_image.py -i images/persons.jpe -o images/persons_3d.jpe
python mask_rcnn.py 3d_image.py -i images/persons.jpe -o images/persons_3d.jpe -w 30 -d 20

## Referrence

[Mask-RCNN](https://github.com/matterport/Mask_RCNN) and [Mask-RCNN-Shiny](https://github.com/huuuuusy/Mask-RCNN-Shiny)

