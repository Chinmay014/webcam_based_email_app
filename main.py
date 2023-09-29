import cv2
import time

# tune this values for best results
color_threshold = 80
area_threshold  = 10000


first_frame = None
video = cv2.VideoCapture(0)
while True:
    check,frame = video.read()
    # convert to gray scale(less processing)
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # apply gaussian blur
    blurred_gray_frame = cv2.GaussianBlur(gray_frame,(21,21),0)
    # cv2.imshow("My_video",gray_frame)

    if first_frame is None:
        first_frame = gray_frame

    # record the difference in frames
    delta_frame = cv2.absdiff(first_frame,gray_frame)
    # cv2.imshow("My_video",delta_frame)
    print(delta_frame)

    # mask the frame with a threshold value(anything above 80 is 255)
    threshold_frame = cv2.threshold(delta_frame,color_threshold,255,cv2.THRESH_BINARY)[1]
    
    # dilate the frame for better filtering
    dilated_frame = cv2.dilate(threshold_frame,None,iterations=2)

    #draw rectangles around the images
    contours,check = cv2.findContours(dilated_frame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 
    for contour in contours:
        # ignore fake objects with small area
        if cv2.contourArea(contour)<area_threshold:
            continue
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    

    cv2.imshow("My_video",frame)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

video.release()

print(frame)
cv2.imwrite("my_image.png",gray_frame)