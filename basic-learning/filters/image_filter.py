# 多种低通滤波器对椒盐噪声的去噪
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import random

src = cv.imread("image\\lenacolor.png")
# src = cv.imread("20200102_20200102163515_20200102171034_163534.jpg")

# matplotlib 和 opencv 通道相反
# 拆分 合并通道 使plt.imshow()显示正常
# b,g,r = cv.split(src)
# img = cv.merge([r,g,b])
img = src

################################################################
# 添加椒盐噪声
# 随机加黑白像素点
# https://blog.csdn.net/pengpengloveqiaoqiao/article/details/89351038
def sp_noise(image,snr):# snr:噪声比例
    h = image.shape[0]
    w = image.shape[1]
    img1 = image.copy()
    sp = h * w  # 计算图像像素点个数
    NP = int(sp * (1 - snr))  # 计算图像椒盐噪声点个数
    for i in range(NP):
        randx = np.random.randint(1, h - 1)  # 生成一个 1 至 h-1 之间的随机整数
        randy = np.random.randint(1, w - 1)  # 生成一个 1 至 w-1 之间的随机整数
        if np.random.random() <= 0.5:  # np.random.random()生成一个 0 至 1 之间的浮点数
            img1[randx, randy] = 0
        else:
            img1[randx, randy] = 255
    return img1

# 添加高斯噪声
# 给每个像素点加随机高斯分布值
def gauss_noise(image,mu = 0,sigma = 10):
    '''
    :param mu:高斯分布的均值 
    :param sigma: 高斯分布标准差
    :return: img
    '''
    img = image.astype(np.int16)  # 图像默认编码为uin8，转为int16方便将灰度值限定在0-255
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                img[i, j, k] = img[i, j, k] + random.gauss(mu=mu, sigma=sigma)
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    return img

# img1 = sp_noise(img,0.8)
# img = gauss_noise(img,0,50)
# ####################################################################
#
# # img = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
# # 2D卷积 5*5卷积核
# # 计算5*5邻域内的
# kernel = np.ones((5,5),np.float32)/25
# dst = cv.filter2D(img,-1,kernel)
#
# # 均值滤波
# blur = cv.blur(img,(5,5))

# # # 高斯滤波
# blur_g = cv.GaussianBlur(img,ksize=(3,3),sigmaX=0) # sigma为0，根据ksize计算sigma
#                                                    # 公式为：σ=0.3×((ksize−1)×0.5−1)+0.8
# blur_g_5 = cv.GaussianBlur(img,(7,7),0)
# sigma越大，越模糊

# # 中位滤波
# median = cv.medianBlur(img,5)
# # 双边滤波
# blur_bi = cv.bilateralFilter(img,9,75,75)
#
# plt.subplot(231),plt.imshow(img),plt.title('original')
# plt.subplot(232),plt.imshow(dst),plt.title('average')
# plt.subplot(233),plt.imshow(blur),plt.title('blurred')
# plt.subplot(234),plt.imshow(blur_g),plt.title('gaussian blurred')
# plt.subplot(235),plt.imshow(median),plt.title('median blurred')
# plt.subplot(236),plt.imshow(blur_bi),plt.title('biateral blurred')
# plt.show()

# 图像锐化
# 拉普拉斯算子
# 此处使用45度增量的滤波器
kernel_laplace = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.float32)
kernel_laplace_2 = np.array([[0,1,0],[1,-4,1],[0,1,0]],np.float32)
0
img_lap_edge = cv.filter2D(img,-1,kernel_laplace)
img_lap_edge_2 = cv.filter2D(img,-1,kernel_laplace_2)
# 滤波器中心为负，需用原图减去变换后的图像，反之为加运算
img_lap = cv.bitwise_and(img,cv.bitwise_not(img_lap_edge))
img_lap_2 = cv.bitwise_and(img,cv.bitwise_not(img_lap_edge_2))

plt.subplot(231),plt.imshow(img),plt.title('original')
plt.subplot(234),plt.imshow(img_lap_edge),plt.title('laplace_-8')
plt.subplot(232),plt.imshow(img_lap),plt.title('changed_-8')
plt.subplot(233),plt.imshow(img_lap_2),plt.title('changed_-4')
plt.subplot(235),plt.imshow(img_lap_edge_2),plt.title('laplace_-4')

plt.show()
cv.namedWindow("1",cv.WINDOW_NORMAL)
cv.namedWindow("2",cv.WINDOW_NORMAL)
cv.namedWindow("3",cv.WINDOW_NORMAL)

# cv.imshow("ksize_3",blur_g)
# cv.imshow("ksize_7",blur_g_5)
# cv.imshow("3",img_lap_edge)
# cv.waitKey(-1)


