import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

src = cv.imread("20200102_20200102163515_20200102171034_163534.jpg")
img = cv.cvtColor(src,cv.COLOR_BGR2GRAY)

# 全局阈值
ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# Otsu阈值
ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# 高斯滤波再Otsu阈值
blur = cv.GaussianBlur(img,(5,5),0)
ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
images = [img, 0, th1,
img, 0, th2,
blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
 'Original Noisy Image','Histogram',"Otsu's Thresholding",
 'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()
img1 = cv.bitwise_and(img,img,mask = th1)
img2 = cv.bitwise_and(img,img,mask = th2)
img3 = cv.bitwise_and(img,img,mask = th3)
image = [img1,img2,img3]

# height,width = img.shape
# img1 = cv.resize(img1,(int(width/2),int(height/2)),interpolation = cv.INTER_AREA)
# img2 = cv.resize(img2,(int(width/2),int(height/2)),interpolation = cv.INTER_AREA)
# img3 = cv.resize(img3,(int(width/2),int(height/2)),interpolation = cv.INTER_AREA)
cv.namedWindow("global",cv.WINDOW_NORMAL)
cv.imshow("global",img1)
cv.imshow("Otsu",img2)
cv.imshow("Gaussian Otsu",img3)
cv.waitKey(-1)
cv.destroyAllWindows()
