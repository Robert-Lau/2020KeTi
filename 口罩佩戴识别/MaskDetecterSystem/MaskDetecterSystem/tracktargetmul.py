from __future__ import print_function
import sys
import cv2
from random import randint
 
trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


def createTrackerByName(trackerType):
  # Create a tracker based on tracker name
  if trackerType == trackerTypes[0]:
    tracker = cv2.TrackerBoosting_create()
  elif trackerType == trackerTypes[1]: 
    tracker = cv2.TrackerMIL_create()
  elif trackerType == trackerTypes[2]:
    tracker = cv2.TrackerKCF_create()
  elif trackerType == trackerTypes[3]:
    tracker = cv2.TrackerTLD_create()
  elif trackerType == trackerTypes[4]:
    tracker = cv2.TrackerMedianFlow_create()
  elif trackerType == trackerTypes[5]:
    tracker = cv2.TrackerGOTURN_create()
  elif trackerType == trackerTypes[6]:
    tracker = cv2.TrackerMOSSE_create()
  elif trackerType == trackerTypes[7]:
    tracker = cv2.TrackerCSRT_create()
  else:
    tracker = None
    print('Incorrect tracker name')
    print('Available trackers are:')
    for t in trackerTypes:
      print(t)
    
  return tracker


def track_init(frame,box=None):
    global bboxes
    bboxes = []
    global colors
    colors = [] 
    print("Default tracking algoritm is CSRT \n"
        "Available tracking algorithms are:\n")
    for t in trackerTypes:
        print(t)      
 
    trackerType = "CSRT"      
 
    ## Select boxes
    if box==None:#未给定box时
        while True:
            # draw bounding boxes over objects
            # selectROI's default behaviour is to draw box starting from the center
            # when fromCenter is set to false, you can draw box starting from top left corner
            bbox = cv2.selectROI('MultiTracker', frame)
            bboxes.append(bbox)
            colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
            print("Press q to quit selecting boxes and start tracking")
            print("Press any other key to select next object")
            k = cv2.waitKey(0) & 0xFF
            if (k == 113):  # q is pressed
                break
        # OpenCV's selectROI function doesn't work for selecting multiple objects in Python
        # So we will call this function in a loop till we are done selecting all objects
  
    else:
        bboxes=box
        for i in range(0,len(bboxes)):
            colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))            
    
           
    print('Selected bounding boxes {}'.format(bboxes))
 
    ## Initialize MultiTracker
    # There are two ways you can initialize multitracker
    # 1. tracker = cv2.MultiTracker("CSRT")
    # All the trackers added to this multitracker
    # will use CSRT algorithm as default
    # 2. tracker = cv2.MultiTracker()
    # No default algorithm specified
 
    # Initialize MultiTracker with tracking algo
    # Specify tracker type
    # Create MultiTracker object
    global multiTracker
    multiTracker = cv2.MultiTracker_create()
        # Initialize MultiTracker 
    for bbox in bboxes:
        multiTracker.add(createTrackerByName(trackerType), frame, bbox)
    

def tracktargetmul(frame):
    
    rect_array=[]
    success, boxes = multiTracker.update(frame)

    # draw tracked objects
    for i, newbox in enumerate(boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
        rect_array.append((p1,p2))
    # show frame
    cv2.imshow('MultiTracker', frame)
    return rect_array

if __name__ == '__main__':
 
    # Set video to load
    #videoPath = "video/run.mp4"
   
    # Create a video capture object to read videos
    cap = cv2.VideoCapture(0)
 
    # Read first frame
    success, frame = cap.read()
    # quit if unable to read the video file
    if not success:
        print('Failed to read video')
        sys.exit(1)
 
    track_init(frame)#划定roi
    
 
    # Process video and track objects
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
    
        track(frame)#跟踪目标
 
        # quit on ESC button
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break
 