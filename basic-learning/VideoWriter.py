# 将相机拍摄保存为指定视频文件
import cv2 as cv

cap = cv.VideoCapture(0,cv.CAP_DSHOW)
# 定义编解码器，创建VideoWriter对象
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('C:\\Users\\Admin\\Desktop\\output.avi', fourcc, 20.0, (640, 480))# （"输出文件名",指定编码器，指定帧率数量，指定帧率大小）
while cap.isOpened():
 ret, frame = cap.read()
 if not ret:
     print("无法接收帧，正在退出...")
     break
 frame = cv.flip(frame, 0)
 # 写翻转的框架
 out.write(frame)
 cv.imshow('frame', frame)
 if cv.waitKey(1) == ord('q'):
     break
# 完成工作后释放所有内容
cap.release()
out.release()
cv.destroyAllWindows()