import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 计算
# cv.calcHist()
img = cv.imread("image\\barbara.bmp")
# img = cv.imread("image\\lena.bmp",cv.COLOR_BGR2GRAY)
hist = cv.calcHist([img],[0],None,[256],[0,256])

# 绘制直方图
# plt.hist(img.ravel(),256,[0,256]);plt.show()

# 绘制BGR图像各通道，像素强度直方图
color = ('b','g','r')
for i,col in enumerate(color):
    # histr = cv.calcHist([img],[i],None,[256],[0,256])
    histr = cv.calcHist([img],[i],None,[256],[0,255])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()

