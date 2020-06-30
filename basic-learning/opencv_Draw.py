# 在窗口中双击鼠标，绘制给定大小的圆
import cv2 as cv
import numpy as np

# cv.imshow("1",img)
# cv.waitKey(0)

events = [i for i in dir(cv) if 'EVENT' in i]
print(events)

# 定义鼠标回调函数
def draw_circle(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDBLCLK: # 鼠标左键双击
        cv.circle(img,(x,y),100,(255,0,0),5)

# 创建一个黑色的图像
img = np.zeros((512,512,3), np.uint8)
cv.namedWindow("image")
# 绑定窗口的功能
cv.setMouseCallback("image",draw_circle)
while(1):
    cv.imshow('image',img)
    if cv.waitKey(20) & 0xFF == 27:  # ESC
        break

cv.destroyAllWindows()



