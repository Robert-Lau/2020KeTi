# 图像锐化-高通滤波
# 二阶微分：Laplacian
# 一阶微分-梯度：Roberts，Sobel,Prewitt
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

src = cv.imread("2.jpg")
img = cv.cvtColor(src,cv.COLOR_BGR2RGB)
grayimg = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
print(grayimg.dtype)
# Laplacian
dst = cv.Laplacian(grayimg,cv.CV_16S,ksize=3)
# 调用 convertScaleAbs() 函数计算绝对值，并将图像转换为8位图进行显示
Laplacian = cv.convertScaleAbs(dst)
cv.imshow("1",Laplacian)
# cv.waitKey(0)
# plt.imshow(Laplacian),plt.title("Laplacian")
# plt.show()

# Roberts
kernelX = np.array([[-1,0],[0,-1]],dtype = int)
kernelY = np.array([[0,-1],[1,0]],dtype = int)
x = cv.filter2D(grayimg,cv.CV_16S,kernelX)
y = cv.filter2D(grayimg,cv.CV_16S,kernelY)
absX = cv.convertScaleAbs(x)
absY = cv.convertScaleAbs(y)
Roberts = cv.addWeighted(absX,0.5,absY,0.5,0)
cv.imshow("2",absY)
cv.waitKey(0)

# Sobel
