## 2020vision

99% of colorblind people suffer from red-green color vision deficiency, which makes it difficult for them to distinguish the colors on a traffic light. Not only is it an annoyance, not being able to see the colors from a distance can cause life-threatening accidents. In fact, motor vehicle accidents are a leading cause of death worldwide.

In order to combat this issue, we must develop an assistive program that alerts colorblind drivers about traffic light information.

Introducing 2020vision. A novel AI solution that detects traffic lights and sends out alerts to the driver. Our program takes in video, identifies nearby traffic lights, targets its color, and uses text to speech to enunciate the status of the traffic light. We want to implement this solution into a car’s dash cam.



We used a combination of YOLO, OpenCV, and python to build our project. 

First, YOLO object detection is used to pinpoint locations and boundaries of various objects in the frame. This is implemented by this boxes method which returns a list of boxes with which the objects are contained.

Next, we must determine the color of the traffic light. We used opencv’s HSV to create masks of different colors in the image. Since traffic lights usually come in the form of a circle, we will search for mask areas inside the traffic lights that contain radial outlines.

Finally we need to find the traffic light relevant to the driver, which is usually closest to the center of view. Once it is identified, the program plays a sound saying the color of the traffic light. The sound will only play +every few seconds when the traffic light is detected.

In the future, some other features that we plan to implement are detection for dysfunctional traffic lights, distance detection, being able to measure distance from the traffic light and alert accordingly, and optimization of color and object detection. 

With 2020vision, we can extend the use of AI to help the colorblind gain a better sense of the road. Let’s build a safer society together.

![2020vision thumbnail](https://github.com/TevinWang/2020vision/blob/master/thumbnail.png)