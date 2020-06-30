
import cv2
import numpy as np

cap = cv2.VideoCapture("D:\\2020KeTi\\basic-learning\\testVideo001.mp4")
# cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

# Check if camera opened successfully
if not(cap.isOpened()):
    print("Error opening video stream or file")
frameNum = 0

# Read until video is completed
while cap.isOpened():
    global previousFrame
    # Capture frame-by-frame
    ret, frame = cap.read()
    print()
    if frame.shape[0] != 1080:
        continue
    frameNum += 1
    if ret:
        tempFrame = frame
        if frameNum == 1:
            previousFrame = cv2.cvtColor(tempFrame, cv2.COLOR_BGR2GRAY)
            print(111)
        if frameNum >= 2:
            currentFrame = cv2.cvtColor(tempFrame, cv2.COLOR_BGR2GRAY)
            currentFrame = cv2.absdiff(currentFrame,previousFrame)
            median = cv2.medianBlur(currentFrame ,3)
            # img = cv2.imread("E:/chinese_ocr-master/4.png")
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret,threshold_frame = cv2.threshold(currentFrame, 20, 255, cv2.THRESH_BINARY)
            gauss_image = cv2.GaussianBlur(threshold_frame, (3, 3), 0)
            print(222)
            # Display the resulting frame
            cv2.imshow('oringinal', frame)
            cv2.imshow('Frame', currentFrame)
            cv2.imshow('median', median)

            # Press Q on keyboard to  exit
            if cv2.waitKey(33) & 0xFF == 27:
                break
            previousFrame = cv2.cvtColor(tempFrame, cv2.COLOR_BGR2GRAY)

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()
