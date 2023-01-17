import cv2 as cv
import numpy as np

def filtering(image):
	# Filtering white pixels
	l_white = np.array([220, 220, 220])
	u_white = np.array([255, 255, 255])
	white_mask = cv.inRange(image, l_white, u_white)
	white_img = cv.bitwise_and(image, image, mask=white_mask)
	# Filtering yellow pixels
	hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
	l_yellow = np.array([10,90,80])
	u_yellow = np.array([110,255,255])
	yellow_mask = cv.inRange(hsv, l_yellow, u_yellow)
	yellow_img = cv.bitwise_and(image, image, mask=yellow_mask)
	
    # Combining both the above images
	img = cv.addWeighted(white_img, 1., yellow_img, 1., 0.)
	return img

p_left_y = None
p_right_y = None

vid = cv.VideoCapture("challenge.mp4")


fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('predict turn.avi', fourcc, 10.0, (1280,720))

while (vid.isOpened()):
    ret,frame = vid.read()

    if ret == True:
            cv.imshow('original', frame)
    else:
        break
    filter = filtering(frame)
    # cv.imshow('f', filter)
   
    gray = cv.cvtColor(filter , cv.COLOR_BGR2GRAY)
    poly = gray.copy()
    internal_poly = np.array( [[[0,720],[200,700],[600,400],[780,420],[1240,710],[1280,720],[1280,0],[0,0]]],dtype=np.int32)
    cv.fillPoly( poly , internal_poly, (0,0,0) )
    blurred =  cv.GaussianBlur(poly,(3,3),cv.BORDER_DEFAULT)
    edge =  cv.Canny(blurred,150,200)
    retval,threshed = cv.threshold(blurred,150,250,cv.THRESH_BINARY)
    cv.imshow("blurred",blurred)
    cv.imshow("gray",edge)
    cv.imshow("threshed",threshed)
    
    h,w = frame.shape[:2]
    cornerpoints = [ (600,424),(780,424),(1100,660),(240,660)]
    # cornerpoints = np.float32([[200, 720],[1100, 720],[595, 450],[685, 450]])
    # p1 =np.float32([(0,0),(w,0),(w,h),(0,h)])
    p1 = np.float32([[(0,0),(200,0),(200,200),(0,200)]])
    p2=np.float32(cornerpoints)
    h1, w1,s = frame.shape 
    H = cv.getPerspectiveTransform(p2,p1)
    # H =cv.findHomography(p2,p1)
    Hinv = cv.getPerspectiveTransform(p1,p2)
    
    
    
    filter_frame = filtering(frame)
    
    warped = cv.warpPerspective(filter,H,(200,200))
    warped = cv.cvtColor(warped,cv.COLOR_BGR2GRAY)

    blurred2 = cv.GaussianBlur(warped, (3,3), cv.BORDER_DEFAULT)
    retval,thresh_warp = cv.threshold(blurred2,150,255,cv.THRESH_BINARY)

    
    
    
    left_x, left_y = np.where(thresh_warp[:, :100] == 255)
    coeffs = np.polyfit(left_x, left_y, 2)
    left_x = np.arange(70, 200, 1)
    left_y = np.polyval(coeffs, left_x)
    
    if p_left_y is None: p_left_y = left_y
    p_left_y = 0.8*p_left_y + 0.2*left_y
    
    left_pts = np.int0(np.c_[p_left_y, left_x])
    
    right_x, right_y = np.where(thresh_warp[:, 130:] == 255)
    coefs = np.polyfit(right_x, right_y, 3)
    right_x = np.arange(70, 200, 1)
    right_y = np.polyval(coefs, left_x)
    
    if p_right_y is None: p_right_y = right_y
    p_right_y = 0.8*p_right_y + 0.2*right_y
    
    right_pts = np.int0(np.c_[130 + p_right_y, right_x])
    
    new = np.uint8(np.zeros((200, 200, 3))*255)
    cv.polylines(new, [left_pts], False, [0, 0, 255], 4)
    cv.polylines(new, [right_pts], False, [0, 0, 255], 4)
    
    points = np.r_[left_pts, np.flipud(right_pts)]
    cv.fillPoly(new, [points], [0,255, 0])
    
    lanes = cv.warpPerspective(new, Hinv, (w, h), flags = cv.INTER_LINEAR)
    final = np.uint8(0.5*frame.copy() + 0.3*lanes)
    
    
    cv.imshow('filtered', filter)
    cv.imshow('gray',gray)
    cv.imshow('blur', blurred)
    cv.imshow("thresh warp",thresh_warp)
    cv.imshow("final",final)
    if cv.waitKey(25) & 0xFF==ord("d"):
        break

    out.write(final)

vid.release()
out.release()
