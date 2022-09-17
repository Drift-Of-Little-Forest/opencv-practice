# ! python-----sx
# -*- coding:utf-8 -*-
# @FileName  :Target detection.py
# @Time      :2022/9/13 10:18
# @Author    :Wandering boy
import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0)


## 导入coco数据集，里面有各种80多种类别
classesFile = "opencv_data_ku/coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)

"===============引入我们的yolo3模块=========="
## Model Files
#可以去yolo官网自己搜下面相关的文件参数
modelConfiguration = "yolov3-320.cfg"
modelWeights = "opencv_data_ku/yolov3.weights"
#调用函数进行模型的配置和权重的配置
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
#要求网络使用其支持的特定计算后
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
#使用CPU进行计算
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


def findObjects(outputs, img):
    hT, wT, cT = img.shape#输出照片的宽高通道数
    bbox = []
    classIds = []
    confs = []
    #对检测处的结果进行对比处理
    for output in outputs:
        for det in output:
            #可以看一下输出的内容
            scores = det[5:]#这里的意思是输出是某个种类的概率，前五个是框的位置以及置信度
            classId = np.argmax(scores)#找到是最大的种类编号
            confidence = scores[classId]#找到置信度
            if confidence > 0.5:#设立阈值
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                #更新新的框
                bbox.append([x, y, w, h])
                #将索引加入到列表里面去
                classIds.append(classId)
                #置信度加入到创建的列表当中去
                confs.append(float(confidence))
    #对于这个函数，可以在目标检测中筛选置信度低于阈值的，还进行Nms筛选，
    # 至于参数，第一个参数是输入的预测框的尺寸，注意这里得尺寸是预测框左上角和右下角的尺寸，类似yolov3这种最后预测的坐标是中心点和长宽
    #第二个参数是预测中的的置信度得分
    #其实这个函数做了两件事情，对预测框和置信度设定阈值，进行筛选
    indices = cv.dnn.NMSBoxes(bbox, confs, 0.5, 0.6)
    #将框画出来
    for i in indices:
        i = i
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',(x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


while True:
    success, img = cap.read()
    #神经网络的输入图像需要采用称为blob的特定格式。用需要检测的原始图像image构造一个blob图像，
    #对原图像进行像素归一化1 / 255.0，缩放尺寸
    # (320, 320),交换了R与B通道
    blob = cv.dnn.blobFromImage(img, 1 / 255, (320, 320), 1, crop=False)
    #将blob变成网络的输入
    net.setInput(blob)
    #获取神经网络所有层的名称
    layersNames = net.getLayerNames()
    print('所有层：',layersNames)
    #我们不需要改变其他层，只需要找到未连接层，也就是网络的最后一层，在最后一层的基础上进行前向传播就可以了
    print('三个输出层的索引号',net.getUnconnectedOutLayers())
    for i in net.getUnconnectedOutLayers():
        outputNames = [layersNames[i - 1]]
        print('输出层名字',outputNames)
    #前向传播
        outputs = net.forward(outputNames)
        print(outputs)
        #了解每个输出层的形状
        print(outputs[0].shape)
    #调用框函数寻找对象的框
        findObjects(outputs, img)

    cv.imshow('Image', img)
    cv.waitKey(1)
