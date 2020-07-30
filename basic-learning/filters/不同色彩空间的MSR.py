'''
RGB转到HSV,HSI,YCbCr(YUV)，CIElab
使用MSR进行去雾
使用信息熵和灰度直方图进行评价
'''

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


def singleScaleRetinex(img, sigma):

    retinex = np.log10(img) - np.log10(cv.GaussianBlur(img, (0, 0), sigma))
    return retinex

def multiScaleRetinex(img, sigma_list):
    '''

    :param img: 对数域图像
    :param sigma_list:
    :return:
    '''

    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex = retinex + singleScaleRetinex(img, sigma)
        # print(retinex)
    # 此结果在对数域
    MSR_result = retinex / len(sigma_list)
    # print(MSR_result)
    return MSR_result

def simplestColorBalance(brightness_img, low_clip, high_clip):
    '''

    :param img: hsv图像的v分量
    :param low_clip:
    :param high_clip:
    :return: SCB处理之后的亮度分量
    '''
    # 按照一定比例去除最小和最大灰度值
    total = brightness_img.shape[0] * brightness_img.shape[1]  # 像素数
    unique, counts = np.unique(brightness_img, return_counts=True) # 排序输出单通道图像灰度值的唯一值及其出现的次数
    current = 0
    for u, c in zip(unique, counts):
        if float(current) / total < low_clip:
            low_val = u
        if float(current) / total < high_clip:
            high_val = u
        current += c
    # 将大于high_val,和小于low_val的灰度值，标定为这两个值
    brightness_img = np.maximum(np.minimum(brightness_img, high_val), low_val)

    return brightness_img

# 量化出结果
def quatification(MSR_result,flag,low_clip,high_clip,G=192,b=-30):

    if flag == 1:
        # gain/offset
        MSR_result = G * (MSR_result + b)
        # 线性量化
        MSR_result = (MSR_result - np.min(MSR_result)) / \
                     (np.max(MSR_result) - np.min(MSR_result)) * 255
        # 溢出处理
        MSR_result = np.uint8(np.minimum(np.maximum(MSR_result, 0), 255))
        # SCB
        quatification_result = simplestColorBalance(MSR_result, low_clip, high_clip)
    elif flag == 2:
        # 线性量化
        MSR_result = (MSR_result - np.min(MSR_result)) / \
                     (np.max(MSR_result) - np.min(MSR_result)) * 255
        # 溢出处理
        MSR_result = np.uint8(np.minimum(np.maximum(MSR_result, 0), 255))
        # SCB
        quatification_result = simplestColorBalance(MSR_result,low_clip,high_clip)
    elif flag == 3:
        pass

    return quatification_result



# HSV_MSR
def HSV_MSRCR(src,sigma_list,flag,low_clip,high_clip,alpha=125,beta=46):
    '''

    :param src: 待处理图像
    :param sigma_list: 各尺度RETINEX的高斯函数sigma列表
    :param flag: 量化方法标记 flag = 1 ：Gain/offset方法
                            flag = 2 ：simplest color balance
                            flag = 3 : GIMP内嵌实现
    :return:
    '''
    # 防止对数操作出现runtime warning
    # min_nonzero = min(src[np.nonzero(src)]) # np.nonzero()返回数组非零元素的索引
    # src[src == 0] = min_nonzero             # 将0像素赋值为最小非零值
    # src[src == 0] = 1

    # 将图像转到HSV空间
    src_hsv = cv.cvtColor(src,cv.COLOR_BGR2HSV)
    # src_hsv[src_hsv==0] = 1
    h,s,v = cv.split(src_hsv)
    # print(src_hsv[0])
    hsv_msr = np.zeros_like(src_hsv)

    # 对v分量进行MSR
    v_msr_result = multiScaleRetinex(v,sigma_list)
    print(v_msr_result)

    # # color restoration
    # img_sum = np.sum(v)
    # color_restoration = beta * (np.log10(alpha * v) - np.log10(img_sum))
    # v_msrcr_result = color_restoration * v_msr_result
    # print(v_msrcr_result)

    # 量化
    quatification_result = quatification(v_msr_result,flag,low_clip,high_clip,G=192,b=-30)
    # 将图像转回RGB
    hsv_msr[:,:,0] = h
    hsv_msr[:,:,1] = s
    hsv_msr[:,:,2] = quatification_result

    rgb_msr = cv.cvtColor(hsv_msr,cv.COLOR_HSV2BGR)

    return rgb_msr,h,s,v

# def

if __name__ == '__main__':
    src = cv.imread("C:\\Users\\lbw\\Desktop\\test\\16.jpg",1)
    sigma_list = [5,15,255]
    result,h,s,v = HSV_MSRCR(src,sigma_list,1,0.01,0.99)
    # 对比度受限直方图均衡化 CLAHE
    clane = cv.createCLAHE(2,(8,8))
    a = cv.cvtColor(result,cv.COLOR_BGR2GRAY)
    cll = clane.apply(a)
    cv.imshow('src',src)
    cv.imshow('v',v)
    cv.imshow('h',h)
    cv.imshow('s',s)
    cv.imshow('1',result)
    cv.imshow('2',cll)
    cv.waitKey()




