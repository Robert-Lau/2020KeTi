# cv.add()   cv.addWeighted
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# cv图像加法，250+10 = 260 => 255
x = np.uint8([250])
y = np.uint8([10])
print(cv.add(x,y))

# 图像加权融合
img1 = cv.imread("C:\\Users\\Admin\\Desktop\\image/barbara.bmp") # 注意图片的格式，读取时可能报错
img2 = cv.imread("C:\\Users\\Admin\\Desktop\\image/girl.bmp")
dst = cv.addWeighted(img1,0.7,img2,0.3,0)
cv.namedWindow('dst',cv.WINDOW_NORMAL)
cv.imshow('dst',dst)
cv.waitKey(-1)
cv.destroyAllWindows()

# 逻辑操作，位运算
# 在一张图像上放置其他图像，区别于图像相加和图像融合
# 图像相加 改变颜色
# 图像融合 透明效果

# 使用ROI进行按位操作
# 加载两张图片
logo = cv.imread("C:\\Users\Admin\Desktop\image\\add\\lena.bmp")
boat = cv.imread("C:\\Users\Admin\Desktop\image\\add\\boat.bmp")
# logo = cv.resize(logo_origin,(300,300),interpolation=cv.INTER_AREA)
# cv.imwrite("C:\\Users\Admin\Desktop\image\\add\\logo300.png",logo)
# 确定ROI的位置
rows1,cols1,channels1 = logo.shape
rows2,cols2,channels2 = boat.shape
print(logo.shape,boat.shape)
# ROI设置在图像正中间
# 此处注意，叠加图像大小大于被叠加图像会报错，需对叠加图像进行适当压缩
roi = boat[((rows2-rows1)//2):((rows1+rows2)//2),((cols2-cols1)//2):((cols1+cols2)//2)]

## 将logo转为灰度图像
logo2gray = cv.cvtColor(logo,cv.COLOR_BGR2GRAY)

# 创建logo掩码，和相反的掩码
## cv.threshold(src,threshold,max_value,threshold_type)
# 具体阈值类型见https://blog.csdn.net/chenriwei2/article/details/30974273

# 将logo灰度图二值化
ret,mask = cv.threshold(logo2gray,100,255,cv.THRESH_BINARY)
# 得到logo反掩码
mask_inv = cv.bitwise_not(mask)
cv.imshow("mask",mask)
cv.imshow("mask_inv",mask_inv)
# 将roi的logo区域涂黑
boat_bg = cv.bitwise_and(roi,roi,mask = mask_inv)
cv.imshow("boat_bg",boat_bg)
# 将logo中的有用区域抠出来
logo_fg = cv.bitwise_and(logo,logo,mask = mask)
# 将抠出的logo放入roi
dst = cv.add(logo_fg,boat_bg)
# 修改原图像
boat[((rows2-rows1)//2):((rows1+rows2)//2),((cols2-cols1)//2):((cols1+cols2)//2)] = dst
cv.imshow("logoboat",boat)
cv.waitKey(-1)
cv.destroyAllWindows()
