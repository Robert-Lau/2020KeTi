# -*- coding = utf-8 -*-
# @time:2020/7/24 16:07
# Author:lbw
# @File:amsrcr比较.py
# @Software:PyCharm
import numpy as np
import cv2
import time
import math
from matplotlib import pyplot as plt

def singleScaleRetinex(img, sigma):

    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))

    return retinex

def multiScaleRetinex(img, sigma_list):

    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += singleScaleRetinex(img, sigma)

    retinex = retinex / len(sigma_list)

    return retinex

# 色彩恢复
def colorRestoration(img, alpha, beta):

    img_sum = np.sum(img, axis=2, keepdims=True)

    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))

    return color_restoration


def automatedMSR(img, sigma_list,y):
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

def automatedMSRCR(img, sigma_list,alpha,beta,G,b):
    img = np.float64(img) + 1.0  # 防止出现0灰度

    img_retinex = multiScaleRetinex(img, sigma_list)
    img_color = colorRestoration(img,alpha,beta)
    img_msrcr = G * (img_retinex * img_color + b)

    for i in range(img_msrcr.shape[2]):
        unique, count = np.unique(np.int32(img_msrcr[:, :, i] * 100), return_counts=True)
        zero_count = 0
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break

        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        # print(unique[0], unique[-1])
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break
        # print(low_val, high_val)
        # 自动裁剪
        img_msrcr[:, :, i] = np.maximum(np.minimum(img_msrcr[:, :, i], high_val), low_val)
        # 映射到0-255
        img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / \
                               (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) \
                               * 255

    img_amsrcr = np.uint8(img_msrcr)

    return img_amsrcr



if __name__ == '__main__':
    sigma_list = [15,80,250]
    alpha,beta = 125.0,46.0
    G,b = 192.0,-30.0
    src = cv2.imread("C:\\Users\\lbw\\Desktop\\test\\18.jpg", 1)
    # img_amsr_1 = automatedMSR(src,sigma_list,0.1)
    img_amsr_05 = automatedMSR(src,sigma_list,0.05)
    B,g,r = cv2.split(img_amsr_05)
    img_amsr_05 = cv2.merge([r,g,B])
    img_amsr_1 = automatedMSR(src,sigma_list,0.1)
    B,g,r = cv2.split(img_amsr_1)
    img_amsr_1= cv2.merge([r,g,B])
    plt.subplot(121),plt.imshow(img_amsr_1),plt.title('amsr1')
    plt.subplot(122),plt.imshow(img_amsr_05),plt.title('amsr05')
    plt.show()
    # res1 = get_entropy(img_amsr)
    # res2 = get_entropy(img_amsrcr)
    # print(res1,res2)
    # cv2.imshow("src",src)
    # cv2.imshow("amsrcr",img_amsrcr)
    # cv2.imshow("amsr05",img_amsr_05)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()




