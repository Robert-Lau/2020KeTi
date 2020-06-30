import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 图片路径必须为中文
img = cv.imread("C:\\Users\\Admin\\Desktop\\image\\lenacolor.png",cv.IMREAD_GRAYSCALE)
cv.imshow("ori",img)

# 缩放cv.resize()
res = cv.resize(img,None,fx=2,fy=2,interpolation=cv.INTER_CUBIC)
cv.imshow("res",res)


# 平移cv.warpAffine() 2*3转换矩阵
rows,cols = img.shape
M = np.float32([[1,0,100],[0,1,50]])
dst = cv.warpAffine(img,M,(cols,rows)) # 第三个参数为输出图像大小（width，height）
cv.imshow('img',dst)

# 旋转cv.getRotationMatrix2D()
RM = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
dst_R = cv.warpAffine(img,RM,(cols,rows))
cv.imshow("rota",dst_R)

# 仿射变换cv.getAffineTransform（）
# 选择原图像中三个点，以及在输出图像中的位置
rows,cols = img.shape
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
A_M = cv.getAffineTransform(pts1,pts2)
dst_A = cv.warpAffine(img,A_M,(cols,rows))
cv.imshow("affine",dst_A)


# 透视变换cv.getPerspectiveTransform()
rows,cols = img.shape
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])
P_M = cv.getPerspectiveTransform(pts1,pts2)
dst = cv.warpPerspective(img,P_M,(cols,rows))
plt.subplot(121),plt.imshow(img,'gray'),plt.title('Input')
plt.subplot(122),plt.imshow(dst,'gray'),plt.title('Output')
plt.show()
cv.waitKey(-1)
cv.destroyAllWindows()


