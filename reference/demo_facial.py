import cv2

# 图片名
filename = 'E:\\cv_pre2seg\\source\\facial2.jpg'


def detect(filename):
    # cv2级联分类器CascadeClassifier,xml文件为训练数据
    face_cascade = cv2.CascadeClassifier('E:\\facial\\opencv\\data\\haarcascades\\haarcascade_frontalface_default.xml')
    # 读取图片
    img = cv2.imread(filename)
    # 转灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 进行人脸检测
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # 绘制人脸矩形框
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # 命名显示窗口
    cv2.namedWindow('people')
    # 显示图片
    cv2.imshow('people', img)
    # 保存图片
    # cv2.imwrite('cxks.png', img)
    # 设置显示时间,0表示一直显示
    cv2.waitKey(0)

detect(filename)
