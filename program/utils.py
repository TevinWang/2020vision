# Modified by Canis and Tevin at TaroHacks: 8/6/2020

# We removed matplotlib in order to continously run frames through opencv, and speed up the process.

# Utils.py originally from here:
# https://github.com/Garima13a/YOLO-Object-Detection



import time
import torch
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches


def boxes_iou(box1, box2):
  
    # Get the Width and Height of each bounding box
    width_box1 = box1[2]
    height_box1 = box1[3]
    width_box2 = box2[2]
    height_box2 = box2[3]
    
    # Calculate the area of the each bounding box
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2
    
    # Find the vertical edges of the union of the two bounding boxes
    mx = min(box1[0] - width_box1/2.0, box2[0] - width_box2/2.0)
    Mx = max(box1[0] + width_box1/2.0, box2[0] + width_box2/2.0)
    
    # Calculate the width of the union of the two bounding boxes
    union_width = Mx - mx
    
    # Find the horizontal edges of the union of the two bounding boxes
    my = min(box1[1] - height_box1/2.0, box2[1] - height_box2/2.0)
    My = max(box1[1] + height_box1/2.0, box2[1] + height_box2/2.0)    
    
    # Calculate the height of the union of the two bounding boxes
    union_height = My - my
    
    # Calculate the width and height of the area of intersection of the two bounding boxes
    intersection_width = width_box1 + width_box2 - union_width
    intersection_height = height_box1 + height_box2 - union_height
   
    # If the the boxes don't overlap then their IOU is zero
    if intersection_width <= 0 or intersection_height <= 0:
        return 0.0

    # Calculate the area of intersection of the two bounding boxes
    intersection_area = intersection_width * intersection_height
    
    # Calculate the area of the union of the two bounding boxes
    union_area = area_box1 + area_box2 - intersection_area
    
    # Calculate the IOU
    iou = intersection_area/union_area
    
    return iou


def nms(boxes, iou_thresh):
    
    # If there are no bounding boxes do nothing
    if len(boxes) == 0:
        return boxes
    
    # Create a PyTorch Tensor to keep track of the detection confidence
    # of each predicted bounding box
    det_confs = torch.zeros(len(boxes))
    
    # Get the detection confidence of each predicted bounding box
    for i in range(len(boxes)):
        det_confs[i] = boxes[i][4]

    # Sort the indices of the bounding boxes by detection confidence value in descending order.
    # We ignore the first returned element since we are only interested in the sorted indices
    _,sortIds = torch.sort(det_confs, descending = True)
    
    # Create an empty list to hold the best bounding boxes after
    # Non-Maximal Suppression (NMS) is performed
    best_boxes = []
    
    # Perform Non-Maximal Suppression 
    for i in range(len(boxes)):
        
        # Get the bounding box with the highest detection confidence first
        box_i = boxes[sortIds[i]]
        
        # Check that the detection confidence is not zero
        if box_i[4] > 0:
            
            # Save the bounding box 
            best_boxes.append(box_i)
            
            # Go through the rest of the bounding boxes in the list and calculate their IOU with
            # respect to the previous selected box_i. 
            for j in range(i + 1, len(boxes)):
                box_j = boxes[sortIds[j]]
                
                # If the IOU of box_i and box_j is higher than the given IOU threshold set
                # box_j's detection confidence to zero. 
                if boxes_iou(box_i, box_j) > iou_thresh:
                    box_j[4] = 0
                    
    return best_boxes


def detect_objects(model, img, iou_thresh, nms_thresh):
    
    # Start the time. This is done to calculate how long the detection takes.
    start = time.time()
    
    # Set the model to evaluation mode.
    model.eval()
    
    # Convert the image from a NumPy ndarray to a PyTorch Tensor of the correct shape.
    # The image is transposed, then converted to a FloatTensor of dtype float32, then
    # Normalized to values between 0 and 1, and finally unsqueezed to have the correct
    # shape of 1 x 3 x 416 x 416
    img = torch.from_numpy(img.transpose(2,0,1)).float().div(255.0).unsqueeze(0)
    
    # Feed the image to the neural network with the corresponding NMS threshold.
    # The first step in NMS is to remove all bounding boxes that have a very low
    # probability of detection. All predicted bounding boxes with a value less than
    # the given NMS threshold will be removed.
    list_boxes = model(img, nms_thresh)
    
    # Make a new list with all the bounding boxes returned by the neural network
    boxes = list_boxes[0][0] + list_boxes[1][0] + list_boxes[2][0]
    
    # Perform the second step of NMS on the bounding boxes returned by the neural network.
    # In this step, we only keep the best bounding boxes by eliminating all the bounding boxes
    # whose IOU value is higher than the given IOU threshold
    boxes = nms(boxes, iou_thresh)
    
    # Stop the time. 
    finish = time.time()
    
    # Print the time it took to detect objects
    print('\n\nIt took {:.3f}'.format(finish - start), 'seconds to detect the objects in the image.\n')
    
    # Print the number of objects detected
    print('Number of Objects Detected:', len(boxes), '\n')
    
    return boxes


def load_class_names(namesfile):
    
    # Create an empty list to hold the object classes
    class_names = []
    
    # Open the file containing the COCO object classes in read-only mode
    with open(namesfile, 'r') as fp:
        
        # The coco.names file contains only one object class per line.
        # Read the file line by line and save all the lines in a list.
        lines = fp.readlines()
    
    # Get the object class names
    for line in lines:
        
        # Make a copy of each line with any trailing whitespace removed
        line = line.rstrip()
        
        # Save the object class name into class_names
        class_names.append(line)
        
    return class_names


def print_objects(boxes, class_names):    
    print('Objects Found and Confidence Level:\n')
    for i in range(len(boxes)):
        box = boxes[i]
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%i. %s: %f' % (i + 1, class_names[cls_id], cls_conf))

            
def plot_boxes(img, cv2, boxes, class_names, plot_labels, color = None):
    
    # Define a tensor used to set the colors of the bounding boxes
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]])
    
    # Define a function to set the colors of the bounding boxes
    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(np.floor(ratio))
        j = int(np.ceil(ratio))
        
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        
        return int(r * 255)
    
    # Get the width and height of the image
    width = img.shape[1]
    height = img.shape[0]
    
    # Create a figure and plot the image
    # fig, a = plt.subplots(1,1)
    # cv2.imshow(img)

    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Plot the bounding boxes and corresponding labels on top of the image
    for i in range(len(boxes)):
        

        # Get the ith bounding box
        box = boxes[i]
        
        

        # Get the (x,y) pixel coordinates of the lower-left and lower-right corners
        # of the bounding box relative to the size of the image. 
        x1 = int(np.around((box[0] - box[2]/2.0) * width))
        y1 = int(np.around((box[1] - box[3]/2.0) * height))
        x2 = int(np.around((box[0] + box[2]/2.0) * width))
        y2 = int(np.around((box[1] + box[3]/2.0) * height))
        
        # Set the default rgb value to red
        rgb = (1, 0, 0)
            
        # Use the same color to plot the bounding boxes of the same object class
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            classes = len(class_names)
            offset = cls_id * 123457 % classes
        
                
            red   = get_color(2, offset, classes) / 255
            green = get_color(1, offset, classes) / 255
            blue  = get_color(0, offset, classes) / 255
            
            # If a color is given then set rgb to the given color instead
            if color is None:
                rgb = (red, green, blue)
            else:
                rgb = color
        # lower_red1 = np.array([0,100,100])
        # upper_red1 = np.array([10,255,255])
        # lower_red2 = np.array([160,100,100])
        # upper_red2 = np.array([180,255,255])
        # lower_green = np.array([40,50,50])
        # upper_green = np.array([90,255,255])
        # # lower_yellow = np.array([15,100,100])
        # # upper_yellow = np.array([35,255,255])
        # lower_yellow = np.array([15,150,150])
        # upper_yellow = np.array([35,255,255])
        # mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        # mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        # maskg = cv2.inRange(hsv, lower_green, upper_green)
        # masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # maskr = cv2.add(mask1, mask2)
        # # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # # Calculate the width and height of the bounding box relative to the size of the image.
        # width_x = x2 - x1
        # width_y = y1 - y2

        # size = img.shape

        font = cv2.FONT_HERSHEY_PLAIN

        # r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1, 80,
        #                        param1=50, param2=10, minRadius=0, maxRadius=30)

        # g_circles = cv2.HoughCircles(maskg, cv2.HOUGH_GRADIENT, 1, 60,
        #                             param1=50, param2=10, minRadius=0, maxRadius=30)

        # y_circles = cv2.HoughCircles(masky, cv2.HOUGH_GRADIENT, 1, 30,
        #                             param1=50, param2=5, minRadius=0, maxRadius=30)

        # # traffic light detect
        # r = 5
        # bound = 4.0 / 10
        # if r_circles is not None:
        #     r_circles = np.uint16(np.around(r_circles))
            
        #     for i in r_circles[0, :]:
        #         if i[0] > size[1] or i[1] > size[0]or i[1] > size[0]*bound:
        #             continue

        #         h, s = 0.0, 0.0
        #         for m in range(-r, r):
        #             for n in range(-r, r):

        #                 if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
        #                     continue
        #                 h += maskr[i[1]+m, i[0]+n]
        #                 s += 1
        #         if h / s > 50:
        #             print('checking r')
        #             # if class_names[cls_id] == "traffic light" and i[0] >= (x1-20) and i[0] <= (x2+20) and i[1] <=(y2+20) and i[1]>=(y1-20):
        #                 # lights.append(Lights(i[0], i[1], "RED", x_center, y_center))
        #             cv2.circle(img, (i[0], i[1]), i[2]+10, (0, 0, 255), 2)
        #             cv2.circle(maskr, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
        #             cv2.putText(img,'RED',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)

        # if g_circles is not None:
        #     g_circles = np.uint16(np.around(g_circles))

        #     for i in g_circles[0, :]:
        #         if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
        #             continue

        #         h, s = 0.0, 0.0
        #         for m in range(-r, r):
        #             for n in range(-r, r):

        #                 if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
        #                     continue
        #                 h += maskg[i[1]+m, i[0]+n]
        #                 s += 1
        #         if h / s > 100:
        #             print('checking g')
        #             # if class_names[cls_id] == "traffic light" and i[0] >= x1 and i[0] <= x2 and i[1] <= y2 and i[1] >= y1:
                        
        #             # lights.append(Lights(i[0], i[1], "GREEN", x_center, y_center))
        #             cv2.circle(img, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
        #             cv2.circle(maskg, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
        #             cv2.putText(img,'GREEN',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)

        # if y_circles is not None:
        #     y_circles = np.uint16(np.around(y_circles))
        #     print('checking y')
        #     for i in y_circles[0, :]:
        #         if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
        #             continue

        #         h, s = 0.0, 0.0
        #         for m in range(-r, r):
        #             for n in range(-r, r):

        #                 if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
        #                     continue
        #                 h += masky[i[1]+m, i[0]+n]
        #                 s += 1
        #         if h / s > 50:
        #                 # if class_names[cls_id] == "traffic light" and i[0] >= x1 and i[0] <= x2 and i[1] <=y2 and i[1]>=y1:
        #                     # lights.append(Lights(i[0], i[1], "YELLOW", x_center, y_center))
        #             cv2.circle(img, (i[0], i[1]), i[2]+10, (0, 255, 255), 2)
        #             cv2.circle(masky, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
        #             cv2.putText(img,'YELLOW',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)
            
        # # Set the postion and size of the bounding box. (x1, y2) is the pixel coordinate of the
        # # lower-left corner of the bounding box relative to the size of the image.
        # # rect = patches.Rectangle((x1, y2),
        # #                          width_x, width_y,
        # #                          linewidth = 2,
        # #                          edgecolor = rgb,
        # #                          facecolor = 'none')

        # # Draw the bounding box on top of the image
        # # if class_names[cls_id] == "traffic light":
        # #     a.add_patch(rect)
        
        # # If plot_labels = True then plot the corresponding label
        
        if plot_labels:
        
            # Create a string with the object class name and the corresponding object class probability
            conf_tx = class_names[cls_id] + ': {:.1f}'.format(cls_conf)
            
            # Define x and y offsets for the labels
            lxc = (img.shape[1] * 0.266) / 100
            lyc = (img.shape[0] * 1.180) / 100
            
            cv2.rectangle(img, (x1,y1), (x2, y2),  (255,0,0), 2)
            cv2.putText(img, conf_tx, (int(x1 + lxc), int(y1 - lyc)), font, 1, (255,0,0), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
                # # Draw the labels on top of the image
                # a.text(x1 + lxc, y1 - lyc, conf_tx, fontsize = 24, color = 'k',
                #     bbox = dict(facecolor = rgb, edgecolor = rgb, alpha = 0.8))        
    
    # print('image shown')    
    # plt.show(block=True)
    # print('afterimage')
    
