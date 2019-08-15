# Multi-Camera-Object-Tracking
Track an object across a non-overlapping CCTV Network.

[DEMO](https://youtu.be/8q7Zv_42oH0)

## Dependencies
+ Directions in /DEPENDENCIES folder
+ opencv and opencv_contrib
+ numpy
+ PyTorch
+ matplotlib

## How it works

When a tracking subject is selected by the user, they are "remembered" by the algorithm. While in the view of a single camera, the subject is tracked using a general object tracking algorithm from OpenCV. When the subject has left the view of a given camera, the surrounding cameras are searched for the tracking subject using the user's initial selection as a reference. Once the algorithm identifies the tracking subject in new camera, single camera tracking resumes.

## Flowchart
[Algorithm Flowchart](Images/algorithm-flowchart.jpg)

