import cv2 as cv
import numpy as np


#reading the video
vid = cv.VideoCapture('whiteline.mp4')


#saving the output
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('Lane Detection.avi', fourcc, 30.0, (960,540))


def draw_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(blank_image, (x1, y1), (x2, y2), (0, 0, 0), 2)

    return img

while (vid.isOpened()):
    ret, frame = vid.read()
    

    if ret == True:
            cv.imshow('original', frame)
    else:
        break
    #converting to grayscale
    gray = cv.cvtColor(frame , cv.COLOR_BGR2GRAY)
    #blurredring
    blurredred = cv.GaussianBlur(gray,(9,9),cv.BORDER_DEFAULT)
    #canny edge detection
    edges = cv.Canny(blurredred, 200, 200)
    #masking the image using a triangular mask
    height, width, _ = frame.shape
    triangle = np.array([[(100, height), (475, 325), (width, height)]])
    mask = np.zeros_like(edges)
    mask = cv.fillPoly(mask, triangle, 255)
    mask = cv.bitwise_and(edges, mask)

    #Probabilistic hough transfor,
    lines = cv.HoughLinesP(mask, rho=2, theta=np.pi/180, threshold=100, minLineLength=2, maxLineGap=10)

    frame_mid_x = frame.shape[1]/2

    left_lane_lines = []
    right_lane_lines = []


    #detecting the lines and classifying them
    for points in lines:
        # print(points)
        x1, y1, x2, y2 = points[0]
        lines_detect = cv.line(frame, (x1, y1), (x2, y2), (0,0,0), 1)

        x_diff = x2 - x1
        y_diff = y2- y1

        if x_diff == 0 and y_diff == 0 :
            continue
    
        slope = y_diff/x_diff
        
        if abs(slope) <= 0.3:
            continue
        if slope < 0 and x1 < frame_mid_x and x2 < frame_mid_x:
            left_lane_lines.append([[x1, y1, x2, y2]])
            cv.line(lines_detect, (x1, y1), (x2, y2), (0, 0, 255 ), 2)

        else:
            right_lane_lines.append([[x1, y1, x2, y2]])
            cv.line(lines_detect, (x1, y1), (x2, y2), (0, 255, 0 ), 2)
            
    right_colored_img = draw_lines(lines_detect, right_lane_lines)



    # cv.imshow('gray', gray)
    # cv.imshow('blurred', blurredred)
    # cv.imshow('canny', edges)
    # cv.imshow('mask', mask)
    cv.imshow('final', right_colored_img)
    if cv.waitKey(20) & 0xFF==ord("d"):
        break
    out.write(right_colored_img)

vid.release()

out.release()
