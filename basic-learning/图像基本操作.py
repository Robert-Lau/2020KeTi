# 像素处理
import cv2
import numpy as np

# 读取像素
# 返回值 = 图像（位置信息）
# opencv彩色图像为BGR图像
img_1 =  cv2.imread("C:\\Users\\lbw\\Desktop\\image\\girl.bmp",cv2.IMREAD_UNCHANGED)
cv2.imshow("demo",img_1)
cv2.waitKey(1000)
p = img_1[100,100] # p[0] p[1] p[2]分别代表B\G\R三个通道
blue = p[0]
green = img_1[100,100,1]
# print(p)
print("blue:" ,blue)
print("green:" ,green)

# 修改像素
## 单一像素
# p = [255,255,255]
# print(p)
## 若干像素
img_1[100:150,100:150] = [255,255,255]
cv2.imshow("change",img_1)

# 使用numpy数组方法进行像素访问与修改
print(img_1.item(50,50,2))   # 访问该位置的R值
img_1.itemset((50,50,2),255) # 修改像素值

# 访问图像属性
print(img_1.shape)  # 按顺序返回列，行和通道数
print(img_1.size)   # 返回像素数
print(img_1.dtype)  # 返回图像数据类型

# 拆分合并图像通道
b,g,r = cv2.split(img_1)
img_1 = cv2.merge([b,g,r])
cv2.imshow("b",img_1)

k = cv2.waitKey(0)
if k == 27: # 等待ESC退出,27为ESC的ASCII码
    cv2.destroyAllWindows()
elif k == ord('s'): # 等待关键字s，保存和退出
    cv2.imwrite('C:\\Users\\Admin\\Desktop\\cube.png',img_1)
    cv2.destroyAllWindows()
