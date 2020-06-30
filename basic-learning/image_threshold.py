# 简单全局阈值分割
# 自适应阈值分割
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 简单阈值cv.threshold(src(graycscale_img),threshold,threshold_maxvalue,threshold_type)
# 返回阈值  和阈值处理后的图像
# 具体阈值类型见https://blog.csdn.net/chenriwei2/article/details/30974273
# src = cv.imread("C:\\Users\Admin\Desktop\image\standard testing image\\bridge.bmp")
# src = cv.imread("C:\\Users\\Admin\\Desktop\image_2\\color\goldhill.png")
src = cv.imread("20200102_20200102163515_20200102171034_163534.jpg")
img = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
ret,thre1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
ret2,thre2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
ret3,thre3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
ret4,thre4 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
ret5,thre5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thre1, thre2, thre3, thre4, thre5]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    # 去除图像坐标显示
    plt.xticks([]),plt.yticks([])
plt.show()

# 自适应阈值
# https://www.cnblogs.com/Anita9002/p/9077472.html
# cv.adaptiveThreshold(src,threshold,自适应阈值方法，threshold_type,像素领域大小，参数c)
# 自适应阈值方法： cv.ADAPTIVE_THRESH_MEAN_C 或 cv.ADAPTIVE_THRESH_GAUSSIAN_C
# 返回阈值处理后的图像
ad_thre1 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,3,3)
ad_thre2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,3,3)
winnames = ['Original Image','Global Thre','Adapative Mean Thre','Adapative Gaussian Thre']
image = [img,thre1,ad_thre1,ad_thre2]
# cv.imshow("1",ad_thre1)
# cv.waitKey(-1)
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(image[i],'gray')
    plt.title(winnames[i])
    plt.xticks([]),plt.yticks([])
plt.show()

