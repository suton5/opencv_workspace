# opencv_workspace

libraries: Contains standard OpenCV libraries
useful: Contains useful Python scripts for various purposes

./useful

1. Extracting frames from camera input
```
python frame_extractor.py
```
Extracts one frame every 5 seconds. Edit file to change delay time/destination folder.

2. Video output
```
python video.py
```
Reads from video source and displays.

./useful/template_matching

1. Template Matching
```
python match.py --template ./template/rmrc1.png --images ./images/rmrc
```
Uses 2D convolution to find a match for the template image in all the images in images folder.
