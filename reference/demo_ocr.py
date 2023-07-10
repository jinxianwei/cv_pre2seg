# text_detection.py 对图像进行文本检测
# text_detection_video 对视频流（实时视频/视频文件）进行文本检测
# frozen_east_text_detection.pb EAST自然场景文本检测序列化的模型
# USAGE
# python text_detection.py --image images/yh.jpg --east frozen_east_text_detection.pb

import argparse
import time

import cv2
import imutils
import numpy as np
# 导入必要的包
from imutils.object_detection import non_max_suppression  # 从IMUTIL中导入了NumPy、OpenCV的非最大单位抑制实现

# 构建命令行参数及解析
# --image 输入图像路径
# --east east场景文本检测器模型路径
# --min-confidence 可选 过滤弱检测的置信度值
# --width 可选 缩放图像宽度，必须是32的倍数
# --height 可选 缩放图像高度，必须是32的倍数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
                help="path to input image")
ap.add_argument("-east", "--east", type=str,
                help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
                help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
                help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
                help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())

# 加载图像获取维度
image = cv2.imread(args["image"])
orig = image.copy()
(H, W) = image.shape[:2]

# 计算宽度，高度及分别的比率值
(newW, newH) = (args["width"], args["height"])
rW = W / float(newW)
rH = H / float(newH)

# 缩放图像获取新维度
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# 为了使用OpenCV和EAST深度学习模型执行文本检测，需要提取两层的输出特征图：
# 定义EAST探测器模型的两个输出层名称，感兴趣的是——第一层输出可能性，第二层用于提取文本边界框坐标
# 第一层是输出sigmoid激活，提供了一个区域是否包含文本的概率。
# 第二层是表示图像“几何体”的输出特征映射-将能够使用该几何体来推导输入图像中文本的边界框坐标
layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]

# cv2.dnn.readNet加载预训练的EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# 从图像构建一个blob，然后执行预测以获取俩层输出结果
# 将图像转换为blob来准备图像
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                             (123.68, 116.78, 103.94), swapRB=True, crop=False)
start = time.time()
# 通过将层名称作为参数提供给网络以指示OpenCV返回感兴趣的两个特征图：
# 分数图，包含给定区域包含文本的概率
# 几何图：输入图像中文本的边界框坐标的输出几何图
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
end = time.time()

# 展示文本预测耗时信息
print("[INFO] text detection took {:.6f} seconds".format(end - start))

# 从分数卷中获取行数和列数，然后初始化边界框矩形集和对应的信心分数
(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []

# 两个嵌套for循环用于在分数和几何体体积上循环，这将是一个很好的例子，说明可以利用Cython显著加快管道操作。
# 我已经用OpenCV和Python演示了Cython在快速、优化的“for”像素循环中的强大功能。
# 遍历预测结果
for y in range(0, numRows):
    # 提取分数（概率），然后是环绕文字的几何（用于推导潜在边界框坐标的数据）
    scoresData = scores[0, 0, y]
    xData0 = geometry[0, 0, y]
    xData1 = geometry[0, 1, y]
    xData2 = geometry[0, 2, y]
    xData3 = geometry[0, 3, y]
    anglesData = geometry[0, 4, y]

    # 遍历列
    for x in range(0, numCols):
        # 过滤弱检测
        if scoresData[x] < args["min_confidence"]:
            continue

        # 计算偏移因子，因为得到的特征图将比输入图像小4倍
        (offsetX, offsetY) = (x * 4.0, y * 4.0)

        # 提取用于预测的旋转角度，然后计算正弦和余弦
        angle = anglesData[x]
        cos = np.cos(angle)
        sin = np.sin(angle)

        # 使用几何体体积导出边界框
        h = xData0[x] + xData2[x]
        w = xData1[x] + xData3[x]

        # 计算文本边界框的开始，结束x，y坐标
        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
        endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
        startX = int(endX - w)
        startY = int(endY - h)

        # 将边界框坐标和概率分数添加到各自的列表
        rects.append((startX, startY, endX, endY))
        confidences.append(scoresData[x])

# 对边界框应用非最大值抑制（non-maxima suppression），以抑制弱重叠边界框，然后显示结果文本预测
# apply overlapping to suppress weak, overlapping bounding boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)

# 遍历边界框
for i, (startX, startY, endX, endY) in enumerate(boxes):
    # 根据相对比率缩放边界框坐标
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)

    # cv2.putText(orig, str(confidences[i]), (startX, startY - 10),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    # 在图像上绘制边界框
    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

# 展示输出图像
cv2.imshow("Text Detection", imutils.resize(orig,width=500))
cv2.waitKey(0)
