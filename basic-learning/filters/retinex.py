'''
此程序有问题，仅作公式参考
'''
import numpy as np
import cv2 as cv

def replaceZeroes(src):
    # 非零像素的最小值
    min_nonzero = min(src[np.nonzero(src)]) # np.nonzero()返回数组非零元素的索引
    src[src == 0] = min_nonzero             # 将0像素赋值为最小非零值
    return src

# 单尺度retinex算法
def SSR(src_img,sigma):
    '''

    :param src_img: 原始图像
    :param sigma: 高斯滤波的方差
    :return:
    '''
    L_blur = cv.GaussianBlur(src_img,(0,0),sigmaX=sigma)  # 照度图像的估计：原图像的高斯模糊
    img = replaceZeroes(src_img)
    L_blur = replaceZeroes(L_blur)

    dst_img = cv.log(img/255.0)
    dst_Lblur = cv.log(L_blur/255.0)
    # ?????????????
    dst_ixl = cv.multiply(dst_img,dst_Lblur)
    log_R = cv.subtract(dst_img,dst_ixl)

    dst_R = cv.normalize(log_R,None,0,255,cv.NORM_MINMAX)
    log_uint8 = cv.convertScaleAbs(dst_R)
    return log_uint8

# 多尺度retinex算法
def MSR(img,scales):
    weight = 1/3.0
    scales_size = len(scales)
    h,w = img.shape[:2]
    log_R = np.zeros((h,w),dtype=np.float32)

    for i in range(scales_size):
        img = replaceZeroes(img)
        L_blur = cv.GaussianBlur(img,(scales[i],scales[i]),0)
        L_blur = replaceZeroes(L_blur)
        dst_img = cv.log(img/255.0)
        dst_Lblur = cv.log(L_blur/255.0)
        dst_iXl = cv.multiply(dst_img,dst_Lblur)
        log_R += weight*cv.subtract(dst_img,dst_Lblur)

    dst_R = cv.normalize(log_R,None,0,255,cv.NORM_MINMAX)
    log_uint8 = cv.convertScaleAbs(dst_R)
    return log_uint8

def simplestColorBalance(img,s1,s2):
    h,w = img.shape[:2]
    temp_img = img.copy()
    one_dim_array = temp_img.flatten() # 按行的方向降维
    sort_array = sorted(one_dim_array) # 升序排列，返回索引列表

    per1 = int((h * w) * s1 / 100)
    minvalue = sort_array[per1]

    per2 = int((h * w) * s2 / 100)
    maxvalue = sort_array[(h * w) - 1 - per2]

    # 实施简单白平衡算法
    if (maxvalue <= minvalue):
        out_img = np.full(img.shape, maxvalue)
    else:
        scale = 255.0 / (maxvalue - minvalue)
        out_img = np.where(temp_img < minvalue, 0)  # 防止像素溢出
        out_img = np.where(out_img > maxvalue, 255)  # 防止像素溢出
        out_img = scale * (out_img - minvalue)  # 映射中间段的图像像素
        out_img = cv.convertScaleAbs(out_img)
    return out_img


def MSRCR(img,scales,s1,s2):
    h,w = img.shape[:2]
    weight = 1/3.0
    alpha = 125.0
    beta = 46.0
    scales_size = len(scales)
    log_R = np.zeros((h,w),dtype=np.float32)

    img_sum = np.sum(img, axis=2, keepdims=True)
    img_sum = replaceZeroes(img_sum)
    gray_img = []

    for i in range(len(img.shape[:2])):
        img[:,:,i] = replaceZeroes(img[:,:,i])
        for j in range(scales_size):
            L_blur = cv.GaussianBlur(img[:,:,i],(scales[i],scales[i]),0)
            L_blur = replaceZeroes(L_blur)

            dst_img = cv.log(img[:,:,i]/255.0)
            dst_Lblur = cv.log(L_blur/255.0)
            dst_ixl = cv.multiply(dst_img,dst_Lblur)
            log_R += weight*cv.subtract(dst_img,dst_ixl)

        MSRCR = beta*(cv.log(alpha*img[:,:,i])-cv.log(img_sum))
        gray = simplestColorBalance(MSRCR,s1,s2)
        gray_img.append(gray)
    return gray_img

if __name__ == '__main__':
    path = "G:/2020KeTi/basic-learning/frogMountain2.jpg"
    src_img = cv.imread(path)
    b,g,r = cv.split(src_img)

    b = SSR(b,3)
    g = SSR(g,3)
    r = SSR(r,3)
    SSR = cv.merge([b,g,r])

    scales = [15,101,301]
    b = MSR(b, scales)
    g = MSR(g, scales)
    r = MSR(r, scales)
    MSR = cv.merge([b, g, r])

    # s1,s2 = 0.01,0.99
    # img = src_img
    # MSRCR = MSRCR(img,scales,s1,s2)

    cv.imshow('original',src_img)
    cv.imshow('SSR',SSR)
    cv.imshow('MSR',MSR)
    # cv.imshow('MSRCR',MSRCR)

    cv.waitKey()
    cv.destroyAllWindows()

