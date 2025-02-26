
Underwater Trash Detection - v5 Dataset_V3_2022-04-21 9:00pm
==============================

This dataset was exported via roboflow.com on April 8, 2024 at 10:14 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 416 images.
Trash-Location are annotated in YOLOv9 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* Equal probability of one of the following 90-degree rotations: none, clockwise

The following transformations were applied to the bounding boxes of each image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Random shear of between -15° to +15° horizontally and -15° to +15° vertically
* Random Gaussian blur of between 0 and 7.75 pixels


