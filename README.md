# Project Vision #
This is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.


## Scaled-YOLOv4: 

* **paper:** https://arxiv.org/abs/2011.08036

* **source code - Pytorch (use to reproduce results):** https://github.com/WongKinYiu/ScaledYOLOv4

* **source code - Darknet:** https://github.com/AlexeyAB/darknet

* **Medium:** https://alexeyab84.medium.com/scaled-yolo-v4-is-the-best-neural-network-for-object-detection-on-ms-coco-dataset-39dfa22fa982?source=friends_link&sk=c8553bfed861b1a7932f739d26f487c8

## YOLOv4:

* **paper:** https://arxiv.org/abs/2004.10934

* **source code:** https://github.com/AlexeyAB/darknet

* **Wiki:** https://github.com/AlexeyAB/darknet/wiki

* **useful links:** https://medium.com/@alexeyab84/yolov4-the-most-accurate-real-time-neural-network-on-ms-coco-dataset-73adfd3602fe?source=friends_link&sk=6039748846bbcf1d960c3061542591d7

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

![scaled_yolov4](https://user-images.githubusercontent.com/4096485/101356322-f1f5a180-38a8-11eb-9907-4fe4f188d887.png) AP50:95 - FPS (Tesla V100) Paper: https://arxiv.org/abs/2011.08036

----

![YOLOv4Tiny](https://user-images.githubusercontent.com/4096485/101363015-e5c21200-38b1-11eb-986f-b3e516e05977.png)

----

![YOLOv4](https://user-images.githubusercontent.com/4096485/90338826-06114c80-dff5-11ea-9ba2-8eb63a7409b3.png)


----

![OpenCV_TRT](https://user-images.githubusercontent.com/4096485/90338805-e5e18d80-dff4-11ea-8a68-5710956256ff.png)












# USING FOR PROJECT VISION - SAHAAY - CFI IITM
(LISTING IMPORTANT COMMANDS RELATED TO VISION FROM https://pjreddie.com/darknet/yolo/)
### Run these commands in linux terminal
-> git clone https://github.com/pjreddie/darknet
-> cd darknet
-> make

## YOLO V3 (NOT TINY)
### run this to download the pretrained weights
-> wget https://pjreddie.com/media/files/yolov3.weights

#### Run the detector on image 'dog.jpg' stored in data folder --- (if image is not provided, it'll prompt you to give one dynamically)
-> ./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg

#### This is a general command for running detection on images
-> ./darknet detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights data/dog.jpg

#### Command for changing threshold hence sensitivity of detector
-> ./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg -thresh 0

## YOLO TINY
### Download these weights
-> wget https://pjreddie.com/media/files/yolov3-tiny.weights

### Run theis detector
-> ./darknet detect cfg/yolov3-tiny.cfg yolov3-tiny.weights data/dog.jpg


## RUNNING ON VIDEO CAM - (aka terrifying part)
#### first compile darknet with cuda and opencv
-> https://pjreddie.com/darknet/install/#cuda

#### once done run this command to switch on cam
-> ./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights

#### SOME MORE INFO
You will need a webcam connected to the computer that OpenCV can connect to or it won't work. 
If you have multiple webcams connected and want to select which one to use you can pass the flag -c <num> to pick (OpenCV uses webcam 0 by default).

You can also run it on a video file if OpenCV can read the video:

-> ./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights <video file>
  
---------------------------------------X--------------------------------------------------------------------
