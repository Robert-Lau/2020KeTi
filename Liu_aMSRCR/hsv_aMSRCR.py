# -*- coding = utf-8 -*-
# @time:2020/7/24 15:50
# Author:lbw
# @File:hsv_aMSRCR.py
# @Software:PyCharm
'''
hsv空间的aMSRCR
'''
import numpy as np
import cv2
import imgEvaluation
import time
import math
from matplotlib import pyplot as plt


# 单通道图像引导滤波
def my_guidedFilter_oneChannel(srcImg, guidedImg, rad=9, eps=0.01):
    srcImg = srcImg / 255.0
    guidedImg = guidedImg / 255.0
    img_shape = np.shape(srcImg)

    #    dstImg=np.zeros(img_shape,dtype=float)
    #
    #    P_mean=np.zeros(img_shape,dtype=float)
    #    I_mean=np.zeros(img_shape,dtype=float)
    #    I_square_mean=np.zeros(img_shape,dtype=float)
    #    I_mul_P_mean=np.zeros(img_shape,dtype=float)
    #    var_I=np.zeros(img_shape,dtype=float)
    #    cov_I_P=np.zeros(img_shape,dtype=float)
    #
    #    a=np.zeros(img_shape,dtype=float)
    #    b=np.zeros(img_shape,dtype=float)
    #    a_mean=np.zeros(img_shape,dtype=float)
    #    b_mean=np.zeros(img_shape,dtype=float)

    P_mean = cv2.boxFilter(srcImg, -1, (rad, rad), normalize=True)
    I_mean = cv2.boxFilter(guidedImg, -1, (rad, rad), normalize=True)

    I_square_mean = cv2.boxFilter(np.multiply(guidedImg, guidedImg), -1, (rad, rad), normalize=True)
    I_mul_P_mean = cv2.boxFilter(np.multiply(srcImg, guidedImg), -1, (rad, rad), normalize=True)

    var_I = I_square_mean - np.multiply(I_mean, I_mean)
    cov_I_P = I_mul_P_mean - np.multiply(I_mean, P_mean)

    a = cov_I_P / (var_I + eps)
    b = P_mean - np.multiply(a, I_mean)

    a_mean = cv2.boxFilter(a, -1, (rad, rad), normalize=True)
    b_mean = cv2.boxFilter(b, -1, (rad, rad), normalize=True)

    dstImg = np.multiply(a_mean, guidedImg) + b_mean

    return dstImg * 255.0

# 三通道图像引导滤波
def my_guidedFilter_threeChannel(srcImg, guidedImg, rad=9, eps=0.01):
    img_shape = np.shape(srcImg)

    dstImg = np.zeros(img_shape, dtype=float)

    for ind in range(img_shape[2]):
        dstImg[:, :, ind] = my_guidedFilter_oneChannel(srcImg[:, :, ind],
                                                       guidedImg[:, :, ind], rad, eps)

    dstImg = dstImg.astype(np.uint8)

    return dstImg

def singleScaleRetinex(img, sigma):

    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
    # retinex = np.log10(img) - np.log10(my_guidedFilter_oneChannel(img, img, 21, 0.01))
    # print(retinex.shape)
    return retinex

def multiScaleRetinex(img, sigma_list):

    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex = retinex + singleScaleRetinex(img, sigma)

    retinex = retinex / len(sigma_list)

    return retinex

# 色彩恢复
def colorRestoration(img, alpha, beta):

    img_sum = np.sum(img, axis=2, keepdims=True)

    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))

    return color_restoration

def simplestColorBalance(img, low_clip, high_clip):

    # 按照一定比例去除最小和最大灰度值
    total = img.shape[0] * img.shape[1]  # 像素数
    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True) # 排序输出单通道图像灰度值的唯一值及其出现的次数
        current = 0
        for u, c in zip(unique, counts):
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c
        # 将大于high_val,和小于low_val的灰度值，标定为这两个值
        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)
        img[:, :, i] = (img[:, :, i] - np.min(img[:, :, i])) / \
                               (np.max(img[:, :, i]) - np.min(img[:, :, i])) \
                               * 255
    return img

def MSRCR(img, sigma_list, G, b, alpha, beta, low_clip, high_clip):

    img = np.float64(img) + 1.0

    img_retinex = multiScaleRetinex(img, sigma_list)   # MSR
    img_color = colorRestoration(img, alpha, beta)     # 色彩恢复：logR = beta*(log(alpha*I)-log(Ir+Ig+Ib) )
    img_msrcr = G * (img_retinex * img_color + b)      # 经典MSRCR论文算法，RMSRCRi(x,y)=G[RMSRCRi(x,y)−b]

    ### 将对数域图像的每个值映射到[0,255]
    for i in range(img_msrcr.shape[2]):
        img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / \
                             (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * \
                             255

    img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))  # 溢出处理
    img_msrcr = simplestColorBalance(img_msrcr, low_clip, high_clip)

    return img_msrcr

def automatedMSRCR(img, sigma_list,y):
    '''
    MSR后自动裁剪
    :param img:
    :param sigma_list:
    :return:
    '''
    zero_count = 0
    img = np.float64(img) + 1.0  # 防止出现0灰度

    img_retinex = multiScaleRetinex(img, sigma_list)

    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break

        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        # print(unique[0], unique[-1])
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * y:
                low_val = u / 100.0
            if u > 0 and c < zero_count * y:
                high_val = u / 100.0
                break
        # print(low_val, high_val)

        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255

    img_retinex = np.uint8(img_retinex)

    return img_retinex


def hsv_automatedMSR(img, sigma_list,y):
    # rgb转hsv
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    zero_count = 0
    # img = np.float64(img) + 1.0  # 防止出现0灰度
    min_nonzero = min(img_hsv[np.nonzero(img_hsv)])  # np.nonzero()返回数组非零元素的索引
    img_hsv[img_hsv == 0] = min_nonzero             # 将0像素赋值为最小非零值
    # img_hsv[img_hsv == 0] = 1
    img_h,img_s,img_v = cv2.split(img_hsv)
    # 对v分量进行msr
    img_v_msr = multiScaleRetinex(img_v, sigma_list)

    # for i in range(img_retinex.shape[2]):
    unique, count = np.unique(np.int32(img_v_msr[:, :] * 100), return_counts=True)
    for u, c in zip(unique, count):
        if u == 0:
            zero_count = c
            break

    low_val = unique[0] / 100.0
    high_val = unique[-1] / 100.0
    # print(unique[0], unique[-1])
    for u, c in zip(unique, count):
        if u < 0 and c < zero_count * y:
            low_val = u / 100.0
        if u > 0 and c < zero_count * y:
            high_val = u / 100.0
            break
    # print(low_val, high_val)

    img_v_msr = np.maximum(np.minimum(img_v_msr, high_val), low_val)

    img_v_msr = (img_v_msr - np.min(img_v_msr)) / \
                           (np.max(img_v_msr) - np.min(img_v_msr)) \
                           * 255
    img_hsv[:,:,0] = img_h
    img_hsv[:,:,1] = img_s
    img_hsv[:,:,2] = img_v_msr
    img_enhanced = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)

    # img_v_msr = np.uint8(img_v_msr)

    return img_enhanced



if __name__ == '__main__':
    sigma_list = [15,80,250]
    alpha, beta = 125.0, 46.0
    G, b = 192.0, -30.0
    # img = cv2.imread("G:\\2020KeTi\\basic-learning\\frogMountain.jpg", 1)
    img = cv2.imread("C:\\Users\\lbw\\Desktop\\test\\18.jpg", 1)

    # img_amsr = automatedMSRCR(img,sigma_list,0.05)

    img_enhanced = hsv_automatedMSR(img,sigma_list,0.05)

    img_msrcr = MSRCR(img,sigma_list,G,b,alpha,beta,0.01,0.99)
    # cv2.imshow('1',img_enhanced)
    # cv2.waitKey()

    # 垂直组合图片，并输出
    res = np.vstack((img,img_enhanced,img_msrcr))
    cv2.imwrite("C:\\Users\\lbw\\Desktop\\gaussain.jpg",res)

    # bgr转rgb，方便pyplot输出
    img = img[...,::-1]
    # img_amsr = img_amsr[...,::-1]
    img_enhanced = img_enhanced[..., ::-1]
    img_msrcr = img_msrcr[...,::-1]

    # 灰度图
    src_ = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_enhanced_ = cv2.cvtColor(img_enhanced,cv2.COLOR_BGR2GRAY)
    img_msrcr_ = cv2.cvtColor(img_msrcr,cv2.COLOR_BGR2GRAY)

    #
    # plt.subplot(321),plt.imshow(src_,'gray'), plt.title('src')
    # plt.subplot(322),plt.hist(src_.ravel(), 256, [0.1, 256])    # 绘制灰度直方图
    # plt.subplot(323),plt.imshow(img_enhanced_,cmap='gray'), plt.title('hsv_amsrcr')
    # plt.subplot(324), plt.hist(img_enhanced_.ravel(), 256, [0.1, 256])
    # plt.subplot(325),plt.imshow(img_msrcr_,'gray'), plt.title('msrcr')
    # plt.subplot(326), plt.hist(img_msrcr_.ravel(), 256, [0.1, 256])
    # plt.show()
    # cv2.waitKey()

    plt.subplot(321), plt.imshow(img), plt.title('src')
    plt.subplot(322), plt.hist(img.ravel(), 256, [0.1, 256])  # 绘制灰度直方图
    plt.subplot(323), plt.imshow(img_enhanced), plt.title('hsv_amsrcr')
    plt.subplot(324), plt.hist(img_enhanced.ravel(), 256, [0.1, 256])
    plt.subplot(325), plt.imshow(img_msrcr), plt.title('msrcr')
    plt.subplot(326), plt.hist(img_msrcr.ravel(), 256, [0.1, 256])
    plt.show()
    cv2.waitKey()

    cv2.destroyAllWindows()

    # 计算信息熵
    imgEvaluation.get_entropy(img)
    imgEvaluation.get_entropy(img_enhanced)
    imgEvaluation.get_entropy(img_msrcr)
    # 计算均值
    imgEvaluation.get_mean(img, 0)
    imgEvaluation.get_mean(img_enhanced, 0)
    imgEvaluation.get_mean(img_msrcr, 0)
    # 计算标准差
    imgEvaluation.get_std(img)
    imgEvaluation.get_std(img_enhanced)
    imgEvaluation.get_std(img_msrcr)
    # 计算平均梯度
    imgEvaluation.get_average_gradient(img)
    imgEvaluation.get_average_gradient(img_enhanced)
    imgEvaluation.get_average_gradient(img_msrcr)
