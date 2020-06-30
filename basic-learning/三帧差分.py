import cv2
import numpy as np

def three_frame_differencing(videopath):
    cap = cv2.VideoCapture(videopath)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    one_frame = np.zeros((height, width), dtype=np.uint8)
    two_frame = np.zeros((height, width), dtype=np.uint8)
    three_frame = np.zeros((height, width), dtype=np.uint8)

    while cap.isOpened():
        ret, frame = cap.read()
        # print(frame.shape[0],frame.shape[1])
        if frame.shape[0]!=1080:
            continue
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # if ((one_frame.shape[0] != 1080)|(two_frame.shape[0]!= 1080)|(three_frame.shape[0]!= 1080)):
        #     continue
        if not ret:
            break
        one_frame, two_frame, three_frame = two_frame, three_frame, frame_gray
        abs1 = cv2.absdiff(one_frame, two_frame)  # 相减
        _, thresh1 = cv2.threshold(abs1, 40, 255, cv2.THRESH_BINARY)  # 二值，大于40的为255，小于0
        abs2 = cv2.absdiff(two_frame, three_frame)
        _, thresh2 = cv2.threshold(abs2, 40, 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_and(thresh1, thresh2)  # 与运算
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        erode = cv2.erode(binary, kernel)  # 腐蚀
        dilate = cv2.dilate(erode, kernel)  # 膨胀
        dilate = cv2.dilate(dilate, kernel)  # 膨胀
        contours, hei = cv2.findContours(dilate.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
        for contour in contours:
            if 100 < cv2.contourArea(contour) < 40000:
                x, y, w, h = cv2.boundingRect(contour)  # 找方框
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
        cv2.namedWindow("binary", cv2.WINDOW_NORMAL)
        cv2.namedWindow("dilate", cv2.WINDOW_NORMAL)
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.imshow("binary", binary)
        cv2.imshow("dilate", dilate)
        cv2.imshow("frame", frame)
        if cv2.waitKey(25) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    three_frame_differencing('D:\\2020KeTi\\basic-learning\\testVideo001.mp4')
    # three_frame_differencing(0)
