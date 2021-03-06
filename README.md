# opencv_workspace
Feel free to contact me at suton5@gmail.com
## libraries: Contains standard OpenCV libraries
## useful: Contains useful Python scripts for various purposes


### ./useful

1. Extracting frames from camera input
```
python frame_extractor.py t
```
Extracts one frame every t seconds. Ensure t is integer/float.

2. Video output
```
python video.py
```
Reads from video source and displays.


### ./useful/template_matching

1. Template Matching
```
python match.py --template ./template/rmrc1.png --images ./images/rmrc
```
Uses 2D convolution to find a match for the template image in all the images in images folder.


### ./useful/haar_cascade/rmrc

1. Haar Cascade classifier training
```
sudo chmod +x train.sh
./train.sh
```
Ensure that there is a folder named 'negatives' with all the images that do not contain the required object (~1000 is good). There should also be a 'pos.png' image that is of the object itself. The convenience script train.sh will create a descriptor file for the negatives, create positive samples by overlaying pos.png onto all the negatives, create a descriptor file for the positives and then train the data.

2. Using the Haar Cascade classifier with a webcam
```
python haar_video.py classifier.xml
```
Reads from video source and finds object. Provide path of classifier as argument.
