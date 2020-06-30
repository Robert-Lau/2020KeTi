# 读 显 存
import cv2

# 读取图像
img_1 =  cv2.imread("C:\\Users\\Admin\\Desktop\\image\\A.png")

# 显示图像
cv2.imshow("demo",img_1) # 一闪而过
cv2.waitKey(-1) # 窗口停留 delay = 0 无限等待
                        #          delay < 0 等待键盘单击打
                        #          delay > 0 等待相应毫秒

# 删除所有窗口（从内存中）
cv2.destroyAllWindows()

# 保存图像
cv2.imwrite("C:\\Users\\Admin\\Desktop\\image\\test.png",img_1)