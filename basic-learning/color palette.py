# 调色板小程序
import numpy as np
import cv2 as cv

def nothing(x):
    pass
# 创建窗口
img = np.zeros((300,512,3),np.uint8)
cv.namedWindow("color_palette")

# 创建调色的轨迹栏
cv.createTrackbar('R','color_palette',0,255,nothing)
cv.createTrackbar('G','color_palette',0,255,nothing)
cv.createTrackbar('B','color_palette',0,255,nothing)

# 创建调色板开关
switch = "0 : OFF \n1 : ON"
cv.createTrackbar(switch,'color_palette',0,1,nothing)
while True:
    cv.imshow('color_palette',img)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    # 四条轨迹栏的当前位置
    r = cv.getTrackbarPos('R','color_palette')
    g = cv.getTrackbarPos('G','color_palette')
    b = cv.getTrackbarPos('B','color_palette')
    s = cv.getTrackbarPos(switch,'color_palette')
    if s == 0:
        img[:] = 0        # 开关关闭，窗口全黑
    else:
        img[:] = [b,g,r]  # 开关打开，根据轨迹栏显示窗口颜色
cv.destroyAllWindows()