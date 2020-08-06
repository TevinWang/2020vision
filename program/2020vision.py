# Created by Canis and Tevin at TaroHacks: 8/6/2020

# This file takes in a video frame by frame, then analyzes it using YOLO object detection and creates boxes.
# Using a HSV color detection model (with parameters optimized for our specific purposes), we added additional
# circles that represented the color of the light.
# We then used the pythagorean theorem to measure distance from the center and alert the driver(using the PlaySound library).

# NOTE: Due to file size, we are unable to upload the REQUIRED yolo weights file, but you can download it here:
# https://pjreddie.com/darknet/yolo/

# DEPENDENCIES:
# - cv2
# - numpy
# - darknet
# - playsound

# Original YOLO example was provided by Garima13a here:
# https://github.com/Garima13a/YOLO-Object-Detection

# HSV color detection formula was originally provided by HevLfreis here:
# https://github.com/HevLfreis/TrafficLight-Detector

import cv2
import numpy as np
import argparse
import time
from darknet import Darknet
from playsound import playsound
import math
from utils import *

# Data about each light needed for alert calculations
class Lights:
    def __init__(self, x, y, color,x_center,y_center):
        self.x = x
        self.y = y
        self.color = color
        self.distCenter  = math.sqrt((x-x_center)**2  + (y-y_center)**2)

    def __str__(self):
        return self.color + " traffic light located at " + str(self.x) + ", " + str(self.y)  + " at a distance of " + str(self.distCenter) + " from the center"

def start_video(video_path):
    # Set the location and name of the cfg file
    cfg_file = './cfg/yolov3.cfg'

    # Set the location and name of the pre-trained weights file
    weight_file = './weights/yolov3.weights'

    # Set the location and name of the COCO object classes file
    namesfile = 'data/coco.names'

    # Set the NMS threshold
    nms_thresh = 0.6

    # Set the IOU threshold
    iou_thresh = 0.4

    class_names = load_class_names(namesfile)

    # Load the network architecture
    m2 = Darknet(cfg_file)

    # Load the pre-trained weights
    m2.load_weights(weight_file)

    # Print the neural network used in YOLOv3
    m2.print_network()

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FPS, 2) 
    video=[]

    while True:
        
        #reading the frame for each video
        ret, frame = cap.read()
       

        # Load the image
        if ret == True:
            # Convert the image to RGB for viewing
            original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            img = frame

            # We resize the image to the input width and height of the first layer of the network.
            resized_image = cv2.resize(original_image, (m2.width, m2.height))


           
            
            cimg = img
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            font = cv2.FONT_HERSHEY_PLAIN
            lights = []
            
            # color ranges for the HSV detection, parameters were optimized by 2020vision

            lower_red1 = np.array([0,100,100])
            upper_red1 = np.array([10,255,255])
            lower_red2 = np.array([160,100,100])
            upper_red2 = np.array([180,255,255])
            lower_green = np.array([40,20,20])
            upper_green = np.array([120,255,255])
            lower_yellow = np.array([15,150,150])
            upper_yellow = np.array([35,255,255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            maskg = cv2.inRange(hsv, lower_green, upper_green)
            
            masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
            maskr = cv2.add(mask1, mask2)
            res = cv2.bitwise_and(frame,frame, mask= mask2)
            cv2.imshow("", res)
            size = img.shape

            # Set the NMS threshold
            nms_thresh = 0.6    

            # Set the IOU threshold
            iou_thresh = 0.4

            width = img.shape[1]
            height = img.shape[0]

            x_center = width / 2
            y_center = height / 2

            # print size
            boxes = detect_objects(m2, resized_image, iou_thresh, nms_thresh)
            print_objects(boxes, class_names)
            # hough circle detect
            r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1, 80,
                                    param1=50, param2=10, minRadius=0, maxRadius=30)

            g_circles = cv2.HoughCircles(maskg, cv2.HOUGH_GRADIENT, 1, 60,
                                        param1=50, param2=10, minRadius=0, maxRadius=30)

            y_circles = cv2.HoughCircles(masky, cv2.HOUGH_GRADIENT, 1, 30,
                                        param1=50, param2=5, minRadius=0, maxRadius=30)


            # traffic light radial outline detection
            r = 5
            bound = 4.0 / 10
            
            # RED MASK
            if r_circles is not None:
            
                r_circles = np.uint16(np.around(r_circles))
                
                for i in r_circles[0, :]:
                    if i[0] > size[1] or i[1] > size[0]or i[1] > size[0]*bound:
                        continue

                    h, s = 0.0, 0.0
                    for m in range(-r, r):
                        for n in range(-r, r):

                            if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                                continue
                            h += maskr[i[1]+m, i[0]+n]
                            s += 1
                    if h / s > 50:
                        for k in range(len(boxes)):
                            
                            # Get the ith bounding box
                            box = boxes[k]
                            if len(box) >= 7 and class_names:
                                cls_conf = box[5]
                                cls_id = box[6]
                                classes = len(class_names)
                                offset = cls_id * 123457 % classes
                        
                            # Get the (x,y) pixel coordinates of the lower-left and lower-right corners
                            # of the bounding box relative to the size of the image. 
                            x1 = int(np.around((box[0] - box[2]/2.0) * width))
                            y1 = int(np.around((box[1] - box[3]/2.0) * height))
                            x2 = int(np.around((box[0] + box[2]/2.0) * width))
                            y2 = int(np.around((box[1] + box[3]/2.0) * height)) #in boxes coordinate

                            if class_names[cls_id] == "traffic light" and i[0] >= (x1-20) and i[0] <= (x2+20) and i[1] <=(y2+20) and i[1]>=(y1-20):
                                lights.append(Lights(i[0], i[1], "RED", x_center, y_center))
                                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 0, 255), 2)
                                cv2.circle(maskr, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                                cv2.putText(cimg,'RED',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)

            # GREEN MASK
            if g_circles is not None:
               
                g_circles = np.uint16(np.around(g_circles))

                for i in g_circles[0, :]:
                    if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                        continue

                    h, s = 0.0, 0.0
                    for m in range(-r, r):
                        for n in range(-r, r):

                            if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                                continue
                            h += maskg[i[1]+m, i[0]+n]
                            s += 1
                    if h / s > 100:
                        for k in range(len(boxes)):
                        
                            # Get the ith bounding box
                            box = boxes[k]

                            if len(box) >= 7 and class_names:
                                cls_conf = box[5]
                                cls_id = box[6]
                                classes = len(class_names)
                                offset = cls_id * 123457 % classes
                            
                            # Get the (x,y) pixel coordinates of the lower-left and lower-right corners
                            # of the bounding box relative to the size of the image. 
                            x1 = int(np.around((box[0] - box[2]/2.0) * width))
                            y1 = int(np.around((box[1] - box[3]/2.0) * height))
                            x2 = int(np.around((box[0] + box[2]/2.0) * width))
                            y2 = int(np.around((box[1] + box[3]/2.0) * height)) #in boxes coordinate

                            #if the circles are detected inside the position of the traffic light, we create a traffic light object,
                            # and add a layer to our computer vision diagram
                            if class_names[cls_id] == "traffic light" and i[0] >= x1 and i[0] <= x2 and i[1] <= y2 and i[1] >= y1:
                                
                                lights.append(Lights(i[0], i[1], "GREEN", x_center, y_center))
                                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                                cv2.circle(maskg, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                                cv2.putText(cimg,'GREEN',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)
            
            # YELLOW MASK
            if y_circles is not None:
                
                y_circles = np.uint16(np.around(y_circles))

                for i in y_circles[0, :]:
                    if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                        continue

                    h, s = 0.0, 0.0
                    for m in range(-r, r):
                        for n in range(-r, r):

                            if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                                continue
                            h += masky[i[1]+m, i[0]+n]
                            s += 1
                    if h / s > 50:
                        for k in range(len(boxes)):
                        
                            # Get the ith bounding box
                            box = boxes[k]
                            if len(box) >= 7 and class_names:
                                cls_conf = box[5]
                                cls_id = box[6]
                                classes = len(class_names)
                                offset = cls_id * 123457 % classes

                            # Get the (x,y) pixel coordinates of the lower-left and lower-right corners
                            # of the bounding box relative to the size of the image. 
                            x1 = int(np.around((box[0] - box[2]/2.0) * width))
                            y1 = int(np.around((box[1] - box[3]/2.0) * height))
                            x2 = int(np.around((box[0] + box[2]/2.0) * width))
                            y2 = int(np.around((box[1] + box[3]/2.0) * height)) #in boxes coordinate

                            if class_names[cls_id] == "traffic light" and i[0] >= x1 and i[0] <= x2 and i[1] <=y2 and i[1]>=y1:
                                lights.append(Lights(i[0], i[1], "YELLOW", x_center, y_center))
                                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 255), 2)
                                cv2.circle(masky, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                                cv2.putText(cimg,'YELLOW',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)
            

            
            # Finding the closest traffic light to the center, and alerting driver.
            minDist = 100000
            minIndex = -1
            for light in range(len(lights)):
                print(str(lights[light]))
                if lights[light].distCenter < minDist:
                    minIndex = light
                    minDist = lights[light].distCenter
            if len(lights) > 0:
                print("Closest traffic light color is " + lights[minIndex].color)
                playsound("./sounds/" + lights[minIndex].color + ".mp3")      
                

            # Plot the image with bounding boxes and corresponding object class labels
            plot_boxes(cimg, cv2, boxes, class_names, plot_labels=True)
            
            # Showing the computer vision imag
            cv2.imshow("Image", cimg)

            # Adding the frame to the output video
            video.append(cimg)
       
            # waiting for a escape key if necessary
            key = cv2.waitKey(100) & 0XFF

          
            
            if key == -1: #no key pressed
                print("continue")
                continue   
            if key == 27: #escape key pressed
                print('escaped')
                break
               
        else:
            break


    # destroying the windows, and writing all video frames to the video
    cv2.destroyAllWindows()
    cap.release()
    height,width,layers=video[0].shape
    video2 = cv2.VideoWriter('../outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (width,height))

    for boximage in range(len(video)):
        video2.write(video[boximage])

   
    #exporting the video
    video2.release()



# add a random video with traffic lights 
if __name__ == '__main__':
    start_video('../example.mp4')
