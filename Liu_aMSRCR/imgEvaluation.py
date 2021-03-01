# -*- coding = utf-8 -*-
# @time:2020/7/28 10:05
# Author:lbw
# @File:imgEvaluation.py
# @Software:PyCharm
'''
图像评价类:
信息熵
均值
标准差
色彩增强因子
'''
import cv2
import numpy as np
import math
import time

# 一维灰度熵计算
# 输入为灰度图
def get_entropy(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        img = img

    x,y = img.shape[0:2]
    tmp = []
    for i in range(256):
        tmp.append(0)
    val = 0
    k = 0
    res = 0
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[val] = float(tmp[val] + 1)
            k = float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if (tmp[i] == 0):
            res = res
        else:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
    print("entropy=",res)

# 均值
def get_mean(img,flag):
    if flag == 1 and len(img.shape) == 3:
        B_mean = np.mean(img[:, :, 0])
        G_mean = np.mean(img[:, :, 1])
        R_mean = np.mean(img[:, :, 2])
        print("bMean=", B_mean, '\n')
        print("gMean=", G_mean, '\n')
        print("rMean=", R_mean, '\n')
    elif flag == 0 and len(img.shape) == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        mean = np.mean(img)
        print('Mean=', mean)
    elif flag == 0 and len(img.shape) == 2:
        mean = np.mean(img)
        print('Mean=', mean)
    elif flag == 1 and len(img.shape) == 2:
        print("图像为灰度图")

# 标准差
def get_std(img):
    if len(img.shape) == 3:
        B_std = np.std(img[:, :, 0])
        G_std = np.std(img[:, :, 1])
        R_std = np.std(img[:, :, 2])
        print("bSTD=",B_std)
        print("gSTD=",G_std)
        print("rSTD=",R_std)
    elif len(img.shape) == 2:
        std = np.std(img)
        print("STD=",std)

# 平均梯度
def get_average_gradient(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = img
    temp = 0
    for i in range(img.shape[0]-1):
        for j in range(img.shape[1]-1):
            dx = np.double(img[i,j+1]) - np.double(img[i,j])
            dy = np.double(img[i+1,j]) - np.double(img[i,j])
            ds = np.sqrt((dx*dx + dy*dy)/2)
            temp += ds

    imgAG = temp/(img.shape[0]*img.shape[1])
    print("AG=",imgAG)

def psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
   print("PSNR=",psnr)

# 对比度
def contrast(img0):
    img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) #彩色转为灰度图片
    m, n = img1.shape
    #图片矩阵向外扩展一个像素
    img1_ext = cv2.copyMakeBorder(img1,1,1,1,1,cv2.BORDER_REPLICATE)
    rows_ext,cols_ext = img1_ext.shape
    b = 0.0
    for i in range(1,rows_ext-1):
        for j in range(1,cols_ext-1):
            b += ((img1_ext[i,j]-img1_ext[i,j+1])**2 + (img1_ext[i,j]-img1_ext[i,j-1])**2 +
                    (img1_ext[i,j]-img1_ext[i+1,j])**2 + (img1_ext[i,j]-img1_ext[i-1,j])**2)

    cg = b/(4*(m-2)*(n-2)+3*(2*(m-2)+2*(n-2))+2*4) #对应上面48的计算公式
    print("contrast=",cg)


