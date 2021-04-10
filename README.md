# CCTVTracker
Track an object across a CCTV Network with non-overlapping camera views.

![ani-gif](https://media.giphy.com/media/RgyqvpsJEbegHCVb08/giphy.gif)

*Chopiness is due to a low framerate camera being used to record CCTV Footage*

[FULL DEMO](https://youtu.be/8q7Zv_42oH0)

### Table of Contents
- <a href='#Dependencies'>Dependencies</a>
- <a href='#How-it-works'>Thought Process</a>
- <a href='#Algorithm-Flowchart'>Flowchart</a>
- <a href='#Output'>Output</a>
- <a href='#Authors'>Authors</a>

&nbsp;
&nbsp;
&nbsp;
&nbsp;

## Dependencies
+ Directions in /DEPENDENCIES folder
+ opencv and opencv_contrib
+ numpy
+ PyTorch
+ matplotlib

## How it works

When a tracking subject is selected by the user, they are "remembered" by the algorithm. While in the view of a single camera, the subject is tracked using a general object tracking algorithm from OpenCV. When the subject has left the view of a given camera, the surrounding cameras are searched for the tracking subject using the user's initial selection as a reference. Once the algorithm identifies the tracking subject in new camera, single camera tracking resumes.

## Algorithm Flowchart
![algorithm-flowchart](https://user-images.githubusercontent.com/21336191/63116742-faadc000-bf5f-11e9-8372-994f0d94395d.jpg)

## Output
The algorithm outputs a video of the tracking subject travelling through the network with a bounding box around them. Below, are still frames from the output video produced on our sample footage.

![cam2](https://user-images.githubusercontent.com/21336191/63117797-1dd96f00-bf62-11e9-8d67-54776a8296dc.jpg)

![cam3](https://user-images.githubusercontent.com/21336191/63117806-22058c80-bf62-11e9-81f1-bc644139a95f.jpg)

### Inputting information about the Camera Network
The algorithm relies on a user-inputted "map" of the camera network (showing relative location and camera adjacency) entered here: 

<img width="800" alt="camera-gui" src="https://user-images.githubusercontent.com/21336191/63117697-ed91d080-bf61-11e9-9676-1dfcf11e6dbf.png">

The user can "place" cameras in placement mode, then indicate the adjacency through the "connections" in connection mode.

## Authors

- [Arvind Ganesh](github.com/arvganesh)
- [Colin Hurley](github.com/colHur)
