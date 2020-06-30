'''
实验：利用单振幅图像/单相位图像/振幅加相位
      进行图像恢复
'''
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 读取灰度图
img = cv.imread('D:\\2020KeTi\\basic-learning\\image\\airfield.bmp',cv.IMREAD_GRAYSCALE)
plt.subplot(221),plt.imshow(img,'gray'),plt.title('original')
# 傅里叶变换
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
# # 绝对值化，加对数
# img_back = np.log(np.abs(fshift))
# plt.subplot(222),plt.imshow(img_back,'gray'),plt.title('only Amplitude')

# 利用振幅进行图像恢复
# 取绝对值即为振幅
# 反中心化
fshift_back = np.fft.ifftshift(np.abs(fshift))
# 反傅里叶变换
img_back = np.fft.ifft2(fshift_back)
img_back = np.abs(img_back)
plt.subplot(222),plt.imshow(img_back,'gray'),plt.title('only Amplitude')

# 利用相位进行图像恢复
# 相位
fshift_angle = np.angle(fshift)
# 反中心化
fshift_angle_back = np.fft.ifftshift(fshift_angle)
# 反傅里叶变换
img_back = np.fft.ifft2(fshift_angle_back)
img_back = np.abs(img_back)
plt.subplot(223),plt.imshow(img_back,'gray'),plt.title('only phase')

# 将振幅和相位结合进行图像恢复
# 振幅
s1 = np.abs(fshift)
# 相位
s1_phase = np.angle(fshift)
# 实部
s1_real = s1*np.cos(s1_phase)
# 虚部
s1_imag = s1*np.sin(s1_phase)
s2 = np.zeros(img.shape,dtype=complex)
s2.real = np.array(s1_real)
s2.imag = np.array(s1_imag)
fshift_2 = np.fft.ifftshift(s2)
img_back = np.abs(np.fft.ifft2(fshift_2))
plt.subplot(224),plt.imshow(img_back,'gray'),plt.title('another way')
plt.show()

