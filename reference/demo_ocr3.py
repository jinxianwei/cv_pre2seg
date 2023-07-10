import cv2
import numpy as np


def match_loc(imgpath):
    model_img = cv2.imread(imgpath)
    test_img = cv2.imread(imgpath)

    b_model, g_model, r_model = cv2.split(model_img)
    b_test, g_test, r_test = cv2.split(test_img)
    w, h = g_model.shape[::-1]

    result = cv2.matchTemplate(g_test, g_model, cv2.TM_CCOEFF_NORMED)
    (min_val, score, min_loc, max_loc) = cv2.minMaxLoc(result)
    bottom_right = (max_loc[0] + w, max_loc[1] + h)
    print('max_loc',max_loc)
    kuang = cv2.rectangle(test_img, max_loc, bottom_right, 255, 2)
    cv2.imshow('test_img',test_img)
    cv2.waitKey(0)
    return test_img,max_loc, g_test

def MSER(test_img,g_test):
    imgcopy = test_img.copy()
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(g_test)  # 获取文本区域
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]  # 绘制文本区域
    cv2.polylines(test_img, hulls, 1, (0, 255, 0))
    cv2.imshow('temp_img', test_img)

    # 将不规则检测框处理成矩形框
    keep = []
    for c in hulls:
        x, y, w, h = cv2.boundingRect(c)
        # 筛选出可能是文本区域的方框，若去掉会出现很多小方框
        # if (h > w * 1.3) or h < 25:
        #     continue
        if h<10:
            continue
        keep.append([x, y, x + w, y + h])
        cv2.rectangle(imgcopy, (x, y), (x + w, y + h), (255, 255, 0), 1)
    cv2.imshow("imgcopy", imgcopy)
    cv2.waitKey(0)
    return keep

def NMS(boxes, overlapThresh):  # NMS 方法（Non Maximum Suppression，非极大值抑制）
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 按得分排序（如没有置信度得分，可按坐标从小到大排序，如右下角坐标）
    idxs = np.argsort(y2)

    # 开始遍历，并删除重复的框
    while len(idxs) > 0:
        # 将最右下方的框放入pick数组
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # 找剩下的其余框中最大坐标和最小坐标
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # 计算重叠面积占对应框的比例，即 IoU
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]

        # 如果 IoU 大于指定阈值，则删除
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("int")

def MSER_NMS(test_img,g_test):
    test_copy = test_img.copy()
    keep = MSER(test_img, g_test)
    keep2 = np.array(keep)
    pick = NMS(keep2, 0.5)
    print("[x] after applying non-maximum, %d bounding boxes" % (len(pick)))
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(test_copy, (startX, startY), (endX, endY), (255, 185, 120), 2)
    cv2.imshow("After NMS", test_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # model_imgpath = 'E:\\cv_pre2seg\\source\\ocr.jpg'
    test_imgpath = 'E:\\cv_pre2seg\\source\\ocr.jpg'
    # model_imgpath = 'E:\\cv_pre2seg\\source\\4.png'
    # test_imgpath = 'E:\\cv_pre2seg\\source\\4.png'
    # model_imgpath = 'E:\\cv_pre2seg\\source\\ocr2.jpg'
    # test_imgpath = 'E:\\cv_pre2seg\\source\\ocr2.jpg'
    test_img,max_loc, g_test = match_loc(test_imgpath)
    MSER_NMS(test_img, g_test)

