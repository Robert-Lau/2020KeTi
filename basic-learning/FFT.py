'''
快速傅里叶变换
numpy实现
opencv实现
'''
import cv2 as cv
import numpy as np
import timeit
from matplotlib import pyplot as plt

# 读取灰度图
img = cv.imread('D:\\2020KeTi\\basic-learning\\image\\airfield.bmp',cv.IMREAD_GRAYSCALE)

######################################
# numpy实现
# 傅里叶变换
f = np.fft.fft2(img)# 结果为复数
print(type(f))
# 中心化，将低频信号放在中间
fshift = np.fft.fftshift(f)
# 取绝对值：将复数变化成实数
# log标定
# 取对数的目的为了将数据变化到较小的范围（比如0-255）
s1 = np.log(np.abs(f))
s2 = np.log(np.abs(fshift))
# plt.subplot(131),plt.imshow(img,'gray'),plt.title('gray')
# plt.subplot(132),plt.imshow(s1,'gray'),plt.title('fft')
# plt.subplot(133),plt.imshow(s2,'gray'),plt.title('fft_center')
# plt.show()

# 相位图像
f_ang = np.angle(f)
fshift_ang = np.angle(fshift)
# plt.subplot(221),plt.imshow(img,'gray'),plt.title('gray')
# plt.subplot(222),plt.imshow(f_ang,'gray'),plt.title('phase')
# plt.subplot(223),plt.imshow(fshift_ang,'gray'),plt.title('phase_shift')

# 复原图像
# 反中心化
fshift_back = np.fft.ifftshift(fshift)
# 反傅里叶
f_back = np.fft.ifft2(fshift_back)
# 绝对值化显示
# plt.subplot(224),plt.imshow(np.abs(f_back),'gray'),plt.title('gray_back')
# plt.show()

###########################################################
# opencv实现
# 输入图像转为float32
dft = cv.dft(np.float32(img),flags = cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
print(type(dft_shift))
magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
plt.subplot(121),plt.imshow(img,'gray'),plt.title('Input Image')
plt.subplot(122),plt.imshow(magnitude_spectrum,'gray'),plt.title('magnitude_spectrum')
plt.show()
###########################################################
# 性能优化
# 使用cv.getOptimalDFTSize()找到最佳大小
# numpy可自动零填充
# opencv需手动零填充
rows,cols = img.shape
print("{} {}".format(rows,cols))
nrows = cv.getOptimalDFTSize(rows)
ncols = cv.getOptimalDFTSize(cols)
print("{} {}".format(nrows,ncols))