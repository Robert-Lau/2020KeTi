# cv.cvtColor cv.inRange
import cv2 as cv
import numpy as np

# 颜色转换 cv.cvtColor(src,flags转换类型)
# 转换类型：cv.COLOR_BGR2GRAY
#           cv.COLOR_BGR2HSV
flags = [i for i in dir(cv) if i.startswith('COLOR_')] # 获取其他转换类型flag
print(flags)

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH,320)   # 获取相机分辨率
cap.set(cv.CAP_PROP_FRAME_HEIGHT,240)

# 下面实现视频中指定颜色阈值物体的跟踪
while True:
    # 读取帧
    ret, frame = cap.read()
    # 转换颜色空间
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    # 定义HSV黑色的范围
    lower_black = np.array([0,0,0])
    upper_black = np.array([180,255,46])
    # 获得指定阈值内的图像掩码
    mask = cv.inRange(hsv,lower_black,upper_black)
    dst = cv.bitwise_and(frame,frame,mask = mask)
    cv.imshow("frame",frame)
    cv.imshow("mask",mask)
    cv.imshow("dst",dst)
    if cv.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()


