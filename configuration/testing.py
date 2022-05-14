import cv2
import numpy as np
 
# Create point matrix get coordinates of mouse click on image
point_matrix = np.zeros((2,2),np.int)
 
counter = 0
def mousePoints(event,x,y,flags,params):
    global counter
    # Left button mouse click event 
    if event == cv2.EVENT_LBUTTONDOWN:
        point_matrix[counter] = x,y
        counter = counter + 1
 
# Read image
img = cv2.imread('ref_image.png')
 
while True:
    for x in range (0,2):
        cv2.circle(img,(point_matrix[x][0],point_matrix[x][1]),3,(0,255,0),cv2.FILLED)
 
    if counter == 2:
        starting_x = point_matrix[0][0]
        starting_y = point_matrix[0][1]
 
        ending_x = point_matrix[1][0]
        ending_y = point_matrix[1][1]
        # Draw line for area of interest
        cv2.line(img, (starting_x, starting_y), (ending_x, ending_y), (0, 255, 0), 3)
 
        # Cropping image
        #img_cropped = img[starting_y:ending_y, starting_x:ending_x]
        #cv2.imshow("ROI", img_cropped)
 
    # Showing original image
    cv2.imshow("Original Image ", img)
    # Mouse click event on original image
    cv2.setMouseCallback("Original Image ", mousePoints)
    # Printing updated point matrix
    print(point_matrix)
    # Refreshing window all time
    cv2.waitKey(1)