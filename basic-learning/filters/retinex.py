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
    weight = 1/len(scales)
    scales_size = len(scales)
    h,w = img.shape[:2]
    log_R = np.zeros((h,w),dtype=np.float32)

    for i in range(scales_size):
        img = replaceZeroes(img)
        L_blur = cv.GaussianBlur(img,scales[i],scales[i],0)
        L_blur = replaceZeroes(L_blur)
        dst_img = cv.log(img/255.0)
        dst_Lblur = cv.log(L_blur)
        dst_iXl = cv.multiply(dst_img,dst_Lblur)
        log_R += weight*cv.subtract(dst_img,dst_iXl)

    dst_R = cv.normalize(log_R,None,0,255,cv.NORM_MINMAX)
    log_uint8 = cv.convertScaleAbs(dst_R)
    return log_uint8



if __name__ == '__main__':
    path = "G:/2020KeTi/basic-learning/frog2.jpg"
    src_img = cv.imread(path)
    b,g,r = cv.split(src_img)
    b = SSR(b,3)
    g = SSR(g,3)
    r = SSR(r,3)
    result = cv.merge([b,g,r])

    cv.imshow('original',src_img)
    cv.imshow('SSR',result)
    cv.waitKey()
    cv.destroyAllWindows()


