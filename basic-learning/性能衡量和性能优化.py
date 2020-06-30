# 函数运行时间和性能优化（有问题）
import cv2 as cv
import time
cv.setUseOptimized(True)
img = cv.imread("C:\\Users\\Admin\\Desktop\\image\\airfield.bmp")
# e1 = cv.getTickCount() # 开始获取时钟周期数
e1 = time.time() # 法二：使用time模块
for i in range(5,49,2):
    img = cv.medianBlur(img,i)
# e2 = cv.getTickCount() # 获取结束时钟周期数e
e2 = time.time()
# t = (e2-e1)/cv.getTickFrequency() # 获取时钟周期频率
t = e2-e1
print(t)

print(cv.useOptimized())
if cv.useOptimized() == True:
    cv.setUseOptimized(False)
print(cv.useOptimized())
e3 = time.time() # 法二：使用time模块
for i in range(5,49,2):
    img = cv.medianBlur(img,i)
e4 = time.time()
t = e4-e3
print(t)

