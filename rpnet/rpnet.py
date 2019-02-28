# Compared to fh0.py
# fh02.py remove the redundant ims in model input
from __future__ import print_function, division
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from torch.autograd import Variable
import numpy as np
import os
import argparse
from time import time
from load_data import *
from roi_pooling import roi_pooling_ims
from torch.optim import lr_scheduler


ap = argparse.ArgumentParser()#声明参数解析器
ap.add_argument("-i", "--images",required=True,#必选
                help="path to the input file")# default="C:/Users/Jet Zhang/Desktop/PR_corner/ccpd_final/home/booy/ccpd_dataset/ccpd_base/"
ap.add_argument("-n", "--epochs", default=10000,
                help="epochs for train")
ap.add_argument("-b", "--batchsize", default=5,
                help="batch size for train")
ap.add_argument("-se", "--start_epoch", default=0,
                help="start epoch for train")#required=True
ap.add_argument("-t", "--test",required=True,
                help="dirs for test")# default="C:/Users/Jet Zhang/Desktop/PR_corner/ccpd_final/home/booy/ccpd_dataset/ccpd_base/"
ap.add_argument("-r", "--resume", default='111',
                help="file for re-train")
ap.add_argument("-f", "--folder", required=True,
                help="folder to store model")#default="./model_store"
ap.add_argument("-w", "--writeFile", default='fh02.out',
                help="file for output")

args = vars(ap.parse_args())#返回对象object的属性和属性值的字典对象。

wR2Path = './model/wR2.pth-best-model'
use_gpu = torch.cuda.is_available()
print ("is use gpu:",use_gpu)

numClasses = 7
numPoints = 4
# classifyNum = 35
imgSize = (480, 480)
# lpSize = (128, 64)
provNum, alphaNum, adNum = 38, 25, 35#alphaNum 第二个字符
batchSize = int(args["batchsize"]) if use_gpu else 6#2不会太小吗
trainDirs = args["images"].split(',')
testDirs = args["test"].split(',')
modelFolder = str(args["folder"]) if str(args["folder"])[-1] == '/' else str(args["folder"]) + '/'#对最后一个字符进行特殊处理
storeName = modelFolder + 'fh02.pth'
if not os.path.isdir(modelFolder):#若不存在，创建文件夹
    os.mkdir(modelFolder)

epochs = int(args["epochs"])
#   initialize the output file
if not os.path.isfile(args['writeFile']):
    with open(args['writeFile'], 'wb') as outF:
        pass


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


class wR2(nn.Module):
    def __init__(self, num_classes=1000):
        super(wR2, self).__init__()
        hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden9 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden10 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self.features = nn.Sequential(
            hidden1,
            hidden2,
            hidden3,
            hidden4,
            hidden5,
            hidden6,
            hidden7,
            hidden8,
            hidden9,
            hidden10
        )
        self.classifier = nn.Sequential(
            nn.Linear(23232, 100),
            # nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            # nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),#num_classes is 4 which refer to the bounding box about left-top and right-down
        )

    def forward(self, x):
        x1 = self.features(x)
        x11 = x1.view(x1.size(0), -1)
        x = self.classifier(x11)
        return x


class fh02(nn.Module):
    def __init__(self, num_points, num_classes, wrPath=None):
        super(fh02, self).__init__()
        self.load_wR2(wrPath)
        self.classifier1 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, provNum),
        )
        self.classifier2 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, alphaNum),
        )
        self.classifier3 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, adNum),
        )
        self.classifier4 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, adNum),
        )
        self.classifier5 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, adNum),
        )
        self.classifier6 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, adNum),
        )
        self.classifier7 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, adNum),
        )

    def load_wR2(self, path):
        self.wR2 = wR2(numPoints)
        if use_gpu:
            self.wR2 = torch.nn.DataParallel(self.wR2, device_ids=range(torch.cuda.device_count()))
        else:
            self.wR2 = torch.nn.DataParallel(self.wR2)

        if not path is None:
            if use_gpu:
                self.wR2.load_state_dict(torch.load(path))
            else:
                self.wR2.load_state_dict(torch.load(path,map_location='cpu'))
            # self.wR2 = self.wR2.cuda()
        # for param in self.wR2.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        x0 = self.wR2.module.features[0](x)
        _x1 = self.wR2.module.features[1](x0)
        x2 = self.wR2.module.features[2](_x1)
        _x3 = self.wR2.module.features[3](x2)
        x4 = self.wR2.module.features[4](_x3)
        _x5 = self.wR2.module.features[5](x4)

        x6 = self.wR2.module.features[6](_x5)
        x7 = self.wR2.module.features[7](x6)
        x8 = self.wR2.module.features[8](x7)
        x9 = self.wR2.module.features[9](x8)
        x9 = x9.view(x9.size(0), -1)
        boxLoc = self.wR2.module.classifier(x9)#(n,4)

        h1, w1 = _x1.data.size()[2], _x1.data.size()[3]
        h2, w2 = _x3.data.size()[2], _x3.data.size()[3]
        h3, w3 = _x5.data.size()[2], _x5.data.size()[3]

        if use_gpu:
            p1 = Variable(torch.FloatTensor([[w1,0,0,0],[0,h1,0,0],[0,0,w1,0],[0,0,0,h1]]).cuda(), requires_grad=False)
            p2 = Variable(torch.FloatTensor([[w2,0,0,0],[0,h2,0,0],[0,0,w2,0],[0,0,0,h2]]).cuda(), requires_grad=False)
            p3 = Variable(torch.FloatTensor([[w3,0,0,0],[0,h3,0,0],[0,0,w3,0],[0,0,0,h3]]).cuda(), requires_grad=False)
        else:
            p1 = Variable(torch.FloatTensor([[w1,0,0,0],[0,h1,0,0],[0,0,w1,0],[0,0,0,h1]]), requires_grad=False)
            p2 = Variable(torch.FloatTensor([[w2,0,0,0],[0,h2,0,0],[0,0,w2,0],[0,0,0,h2]]), requires_grad=False)
            p3 = Variable(torch.FloatTensor([[w3,0,0,0],[0,h3,0,0],[0,0,w3,0],[0,0,0,h3]]), requires_grad=False)

        # x, y, w, h --> x1, y1, x2, y2
        assert boxLoc.data.size()[1] == 4

        if use_gpu:
            postfix = Variable(torch.FloatTensor([[1,0,1,0],[0,1,0,1],[-0.5,0,0.5,0],[0,-0.5,0,0.5]]).cuda(), requires_grad=False)
        else:
            postfix = Variable(torch.FloatTensor([[1,0,1,0],[0,1,0,1],[-0.5,0,0.5,0],[0,-0.5,0,0.5]]), requires_grad=False)

        #(n,4)*(4,4)->(n,4)
        boxNew = boxLoc.mm(postfix).clamp(min=0, max=1)#boxLoc.mm(postfix)为根据cx，cy，w，h计算出left-top和right-buttom的点

        # input = Variable(torch.rand(2, 1, 10, 10), requires_grad=True)
        # rois = Variable(torch.LongTensor([[0, 1, 2, 7, 8], [0, 3, 3, 8, 8], [1, 3, 3, 8, 8]]), requires_grad=False)
        roi1 = roi_pooling_ims(_x1, boxNew.mm(p1), size=(16, 8))#boxNew.mm(p1)得到特征图中对应的left-top和right-buttom点的坐标(n,4)
        roi2 = roi_pooling_ims(_x3, boxNew.mm(p2), size=(16, 8))
        roi3 = roi_pooling_ims(_x5, boxNew.mm(p3), size=(16, 8))
        
        rois = torch.cat((roi1, roi2, roi3), 1)#(n,(c1+c2+c3),h,w)

        _rois = rois.view(rois.size(0), -1)

        y0 = self.classifier1(_rois)#使全连接网络自己去学习到feature map和label的对齐关系
        y1 = self.classifier2(_rois)
        y2 = self.classifier3(_rois)
        y3 = self.classifier4(_rois)
        y4 = self.classifier5(_rois)
        y5 = self.classifier6(_rois)
        y6 = self.classifier7(_rois)

        return boxLoc, [y0, y1, y2, y3, y4, y5, y6]#返回检测和识别的结果


def isEqual(labelGT, labelP):#判断标签是否预测正确
    compare = [1 if int(labelGT[i]) == int(labelP[i]) else 0 for i in range(7)]
    # print(sum(compare))
    return sum(compare)


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
 
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
 
    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
 
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / (sum_area - intersect)


def eval(model, test_dirs):
    count, detectionCorrect, recognitionCorrect = 0, 0, 0
    dst = labelTestDataLoader(test_dirs, imgSize)
    if use_gpu:
        testloader = DataLoader(dst, batch_size=1, shuffle=True, num_workers=8)
    else:
        testloader = DataLoader(dst, batch_size=1, shuffle=True, num_workers=0)
    start = time()
    for i, (XI, corner, labels, ims) in enumerate(testloader):
        count += 1
        #测试识别准确率
        YI = [[int(ee) for ee in el.split('_')[:7]] for el in labels]
        if use_gpu:
            x = Variable(XI.cuda(0))
        else:
            x = Variable(XI)
        # Forward pass: Compute predicted y by passing x to the model

        fps_pred, y_pred = model(x)

        outputY = [el.data.cpu().numpy().tolist() for el in y_pred]
        labelPred = [t[0].index(max(t[0])) for t in outputY]

        # compare YI, outputY
        try:
            if isEqual(labelPred, YI[0]) == 7:
                recognitionCorrect += 1
            else:
                pass
        except:
            pass
        img = cv2.imread(ims[0])

        #测试检测准确率
        [cx, cy, w, h] = fps_pred.data.cpu().numpy()[0].tolist()
        left_up = [(cx - w/2)*img.shape[1], (cy - h/2)*img.shape[0]]
        right_down = [(cx + w/2)*img.shape[1], (cy + h/2)*img.shape[0]]

        prediction = (left_up[1],left_up[0],right_down[1],right_down[0])
        group_truth = (corner[0][1].item(),corner[0][0].item(),corner[1][1].item(),corner[1][0].item())
        IOU = compute_iou(prediction,group_truth)

        #print("iou is:",IOU)
        if(IOU>=0.7):
            detectionCorrect+=1

    return detectionCorrect/count, recognitionCorrect/count, (time() - start) / count



epoch_start = int(args["start_epoch"])
resume_file = str(args["resume"])
#if not resume_file == '111':#在原来的基础上继续训练模型
#    # epoch_start = int(resume_file[resume_file.find('pth') + 3:]) + 1
#    if not os.path.isfile(resume_file):
#        print ("fail to load existed model! Existing ...")
#        exit(0)
#    print ("Load existed model! %s" % resume_file)
#    model_conv = fh02(numPoints, numClasses, wR2Path)
#    if use_gpu:
#        model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
#        model_conv.load_state_dict(torch.load(resume_file))
#        model_conv = model_conv.cuda()
#    else:
#        model_conv = torch.nn.DataParallel(model_conv)
#        model_conv.load_state_dict(torch.load(resume_file,map_location='cpu'))
#
#else:#从头开始训练模型
model_conv = fh02(numPoints, numClasses, wR2Path)
if use_gpu:
    model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
    model_conv = model_conv.cuda()
else:
    model_conv = torch.nn.DataParallel(model_conv)


print(model_conv)#打印模型结构
print(get_n_params(model_conv))#打印参数数量

criterion = nn.CrossEntropyLoss()#采用交叉熵损失函数
# optimizer_conv = optim.RMSprop(model_conv.parameters(), lr=0.01, momentum=0.9)
optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)

#导入数据集
dst = labelFpsDataLoader(trainDirs, imgSize)#(480,480)
if use_gpu:
    trainloader = DataLoader(dst, batch_size=batchSize, shuffle=True, num_workers=8)
else:
    trainloader = DataLoader(dst, batch_size=batchSize, shuffle=True, num_workers=0)

lrScheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)

def train_model(model, criterion, optimizer, num_epochs=25):
    # since = time.time()
    best_detectionAp = 0
    loss_log = []
    for epoch in range(epoch_start, num_epochs):
        print("epoch:",epoch)
        lossAver = []
        model.train(True)
        lrScheduler.step()
        start = time()

        for i, (XI, Y, labels, ims, total) in enumerate(trainloader):
            #test and debug
            # if(i==2):sys.exit(0)

            # print(Y,labels,ims)
            #Y：为左上角和左下角的4个坐标值 
            #labels： 为0_0_8_9_24_30_32
            #ims：图片的路径
            
            if not len(XI) == batchSize:
                continue

            #将标签的字符串记进行切割，存储到list中
            YI = [[int(ee) for ee in el.split('_')[:7]] for el in labels]
            # print("YI:",YI)

            #将x1,y1,x2,y2转变为numpy数组的形式
            Y = np.array([el.numpy() for el in Y]).T
            # print("Y:",Y)

            if use_gpu:
                x = Variable(XI.cuda(0))
                y = Variable(torch.FloatTensor(Y).cuda(0), requires_grad=False)
            else:
                x = Variable(XI)
                y = Variable(torch.FloatTensor(Y), requires_grad=False)
            # Forward pass: Compute predicted y by passing x to the model

            fps_pred, y_pred = model(x)


            # print("fps_pred:",fps_pred)
            # y_pred_list = []
            # for yi in y_pred:
            #     y_pred_list.append(np.argmax(yi.detach().numpy(),1))
            # print("y_pred:",y_pred)
            # print("torch.FloatTensor(np.hsplit(fps_pred.numpy(),2)[0]):",Variable(torch.FloatTensor(np.hsplit(fps_pred.detach().numpy(),2)[0])))
            # Compute and print loss
            detection_loss = 0.0
            if use_gpu:
                detection_loss += 0.8 * nn.L1Loss().cuda()(fps_pred[:,:2], y[:,:2])#cx和cy的权重为0.8，w和h的权重为0.2
                detection_loss += 0.2 * nn.L1Loss().cuda()(fps_pred[:,2:], y[:,2:])#这里应该有问题，这样取的每一组数据的前两维，这样写的话变成取每个batch的前两个样本了
            else:
                # changed by sz
                detection_loss += 0.8 * nn.L1Loss().cuda()(fps_pred[:][:2], y[:][:2])
                detection_loss += 0.2 * nn.L1Loss().cuda()(fps_pred[:][2:], y[:][2:])

            # print("fps_pred[:][:2]:",fps_pred[:][:2])
            # print("y[:][:2]:",y[:][:2])
            # print("loss:",loss)

            classfication_loss = 0.0 

            for j in range(7):
                if use_gpu:
                    l = Variable(torch.LongTensor([el[j] for el in YI]).cuda(0))
                else:
                    l = Variable(torch.LongTensor([el[j] for el in YI]))

                classfication_loss += criterion(y_pred[j], l)#直接将各个部分的

            loss = detection_loss + classfication_loss
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()#反向传播
            loss.backward()
            optimizer.step()#更新参数

            lossAver.append(loss.item())
            if i%500==0:
                print ('epoch:{}[{}/{}]===>train average loss:{} = [detection_loss : {}  + classfication_loss：{} ]  spend time:{}'.format(epoch,i,torch.sum(total)//batchSize//batchSize, sum(lossAver) / len(lossAver), detection_loss,classfication_loss, time()-start))

            
            # if i % 50 == 1:#没50个batch写一次日志
            # with open(args['writeFile'], 'a') as outF:
            #     outF.write('train %s images, use %s seconds, loss %s\n' % (i*batchSize, time() - start, sum(lossAver) / len(lossAver) if len(lossAver)>0 else 'NoLoss'))
            # torch.save(model.state_dict(), storeName)

        # print ('epoch:%s  train average loss:%s spend time:%s\n' % (epoch, sum(lossAver) / len(lossAver), time()-start))
        #每个epoch之后进行evaluation
        # model.eval()
        # count, correct, error, precision, avgTime = eval(model, testDirs)
        # with open(args['writeFile'], 'a') as outF:
        #     outF.write('%s %s %s\n' % (epoch, sum(lossAver) / len(lossAver), time() - start))
        #     outF.write('*** total %s error %s precision %s avgTime %s\n' % (count, error, precision, avgTime))
        # torch.save(model.state_dict(), storeName + str(epoch))#保存模型

        model.eval()
        detectionAp, recognitionAp,avgTime = eval(model, testDirs)
        print('Evaluation epoch:{}====>detectionAp is {} recognitionAp is {} avgTime {}'.format(epoch , detectionAp, recognitionAp,avgTime))

        loss_log.append(sum(lossAver) / len(lossAver))
        
        if(detectionAp>best_detectionAp):
            best_detectionAp = detectionAp
            torch.save(model.state_dict(), storeName +'-best-model')
            print("save the best model in epoch {} with {} precision".format(epoch,best_detectionAp))

        if(epoch%5==0):#每20个epoch存一次模型
            torch.save(model.state_dict(), storeName +'-pre-epoch-'+ str(epoch))#保存模型

    print("The loss information is:",loss_log)
    return model

model_conv = train_model(model_conv, criterion, optimizer_conv, num_epochs=epochs)
