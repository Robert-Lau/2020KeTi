import trackmove as mov
import tracktarget as tar
import tracktargetmul as tarmul
import cv2
import sys

if __name__ == '__main__' :
    #cap = cv2.VideoCapture(0) #参数为0是打开摄像头，文件名是打开视频
    #cap = cv2.VideoCapture("images/vtest.avi") #参数为0是打开摄像头，文件名是打开视频
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    # Read first frame.
    #ok, frame = cap.read()
    #if not ok:
    #    print('Cannot read video file')
    #    sys.exit()
    rect_arr=None
    #tarmul.track_init(frame,rect_arr)#目标跟踪初始化
    while True:
         # Read first frame.
        ok, frame = cap.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()
        # Exit if ESC pressed
        k = cv2.waitKey(30) & 0xff
        if k == 27 : 
            break
        #rect_array=tarmul.tracktargetmul(frame)#追踪多目标
        #print(rect_array[0][0])
        rect_arr=mov.trackmove(frame)#检测运动物体
        if rect_arr!= None:
            print(rect_arr)
            tarmul.track_init(frame,rect_arr)#目标跟踪初始化
            rect_array=tarmul.tracktargetmul(frame)#追踪多目标
    cv2.destroyAllWindows()#关闭所有窗口