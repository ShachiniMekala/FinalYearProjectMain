from tracemalloc import start
import cv2 as cv
import configparser
import numpy as np

#colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
fonts = cv.FONT_HERSHEY_COMPLEX

disp_msg="Frame Captured"
point_matrix = [(0,0),(0,0)]


camera = cv.VideoCapture('../lane1.mp4')
capture = False
 
counter = 0
def mousePoints(event,x,y,flags,params):
    global counter 
    if event == cv.EVENT_LBUTTONDOWN:
        point_matrix[counter] = (x,y)
        counter = counter + 1

while True:
    ret, frame = camera.read()

    img_save=frame.copy()

    if capture == True:
        disp_msg=""
        cv.putText(frame, disp_msg, (30, 30), fonts, 0.6, GREEN, 2)
    else:
        disp_msg="Press 'C' to capture the frame"
        cv.putText(frame, disp_msg, (30, 30), fonts, 0.6, RED, 2)

    cv.imshow('frame', frame)
    key = cv.waitKey(1)

    if key == ord('c'):
        capture = True
        cv.imwrite(f'ref_image.png', img_save)
        
        break

    if key == ord('q'):
        break

disp_msg=""
cv.putText(img_save, disp_msg, (30, 30), fonts, 0.6, RED, 2)

roi=cv.selectROI(img_save)
cropped_img=img_save[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
cv.imwrite(f'ref_image_cropped.png', cropped_img)

cv.destroyAllWindows()

img = cv.imread('ref_image_cropped.png')

#draw line
while True:

    disp_msg = "Select two points on the image | 'q' to exit"

    for x in range (0,2):
        cv.circle(img,point_matrix[x],3,(0,255,0),cv.FILLED)
 
    if counter == 2:

        # Draw line for area selected area
        cv.line(img, point_matrix[0], point_matrix[1], (0, 255, 0), 3)

        cfg = configparser.ConfigParser()

        with open('config.ini', encoding='utf-8') as f:
            cfg.read_file(f)

        cfg['region_of_interest']['roi_0'] = str(int(roi[0]))
        cfg['region_of_interest']['roi_1'] = str(int(roi[1]))
        cfg['region_of_interest']['roi_2'] = str(int(roi[2]))
        cfg['region_of_interest']['roi_3'] = str(int(roi[3]))
        cfg['line_indicator']['point_matrix1'] = str(point_matrix)

        with open('config.ini', 'w', encoding='utf-8') as f:
            cfg.write(f)
    
    cv.imshow("line_selector", img)
    cv.putText(img, disp_msg, (30, 30), fonts, 0.6, RED, 2)
    cv.setMouseCallback("line_selector", mousePoints)

    key = cv.waitKey(1)
    if key == ord('q'):
        break


    



