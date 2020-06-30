import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 均值滤波器
mean_filter = np.ones((3,3))
# 高斯滤波器
x = cv.getGaussianKernel(5,10)
gaussian = x*x.T
# x方向的scharr
scharr = np.array([[-3,0,3],
                  [-10,0,10],
                  [-3,0,3]])
# x方向的sobel
sobel_x= np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
# y方向上的sobel
sobel_y= np.array([[-1,-2,-1],
                   [0, 0, 0],
                   [1, 2, 1]])
# 拉普拉斯变换
laplacian=np.array([[0, 1, 0],
                    [1,-4, 1],
                    [0, 1, 0]])
filters = [mean_filter, gaussian, laplacian, sobel_x, sobel_y, scharr]
filter_name = ['mean_filter', 'gaussian','laplacian', 'sobel_x', \
'sobel_y', 'scharr_x']
fft_filters = [np.fft.fft2(x) for x in filters]
fft_shift = [np.fft.fftshift(y) for y in fft_filters]
spectrum = [np.log(np.abs(z)+1) for z in fft_shift]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(spectrum[i],'gray'),plt.title(filter_name[i])
plt.show()