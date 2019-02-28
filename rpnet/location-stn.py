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
#from roi_pooling import roi_pooling_ims
from torch.optim import lr_scheduler
import torch.nn.functional as F

ap = argparse.ArgumentParser()#声明参数解析器
ap.add_argument("-i", "--images",required=True,#必选
                help="path to the input file")# default="C:/Users/Jet Zhang/Desktop/PR_corner/ccpd_final/home/booy/ccpd_dataset/ccpd_base/"
ap.add_argument("-n", "--epochs", default=500,
                help="epochs for train")
ap.add_argument("-b", "--batchsize", default=64,
                help="batch size for train")
ap.add_argument("-se", "--start_epoch", default=0,
                help="start epoch for train")#required=True
ap.add_argument("-t", "--test",required=True,
                help="dirs for test")# default="C:/Users/Jet Zhang/Desktop/PR_corner/ccpd_final/home/booy/ccpd_dataset/test/"
ap.add_argument("-r", "--resume", default="111",
                help="file for re-train")#default='./model/wR2.pth'
ap.add_argument("-f", "--folder", required=True,
                help="folder to store model")#default="./model_loc_store"
# ap.add_argument("-w", "--writeFile", default='fh02.out',
#                 help="file for output")

args = vars(ap.parse_args())#返回对象object的属性和属性值的字典对象。

wR2Path = './model/wR2.pth'
use_gpu = torch.cuda.is_available()
print ("is use gpu:",use_gpu)

numClasses = 4
numPoints = 4
# classifyNum = 35
imgSize = (480, 480)
# lpSize = (128, 64)
provNum, alphaNum, adNum = 38, 25, 35#alphaNum 第二个字符
batchSize = int(args["batchsize"]) if use_gpu else 4#2不会太小吗
trainDirs = args["images"].split(',')
testDirs = args["test"].split(',')
modelFolder = str(args["folder"]) if str(args["folder"])[-1] == '/' else str(args["folder"]) + '/'#对最后一个字符进行特殊处理
storeName = modelFolder + 'wR2-stn.pth'
if not os.path.isdir(modelFolder):#若不存在，创建文件夹
    os.mkdir(modelFolder)

epochs = int(args["epochs"])
#   initialize the output file
# if not os.path.isfile(args['writeFile']):
#     with open(args['writeFile'], 'wb') as outF:
#         pass


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
        ).cuda()

        self.localization = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=5),#[4, 512, 7, 7]
            # nn.MaxPool2d(2),
            nn.ReLU(True),

            nn.Conv2d(192, 128, kernel_size=3),#torch.Size([4, 512, 7, 7])
            # nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        ).cuda()            
         # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(3200, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        ).cuda()

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x).cuda()#torch.Size([4, 512, 5, 5])
        xs = xs.view(xs.size(0), -1).cuda()#xs.view(-1, 512*7*7)
        theta = self.fc_loc(xs).cuda()
        theta = theta.view(-1, 2, 3).cuda()
        # #
        grid = F.affine_grid(theta, x.size()).cuda()
        x = F.grid_sample(x, grid).cuda()
        return x

    def forward(self, x):
        x1 = self.features(x).cuda()
        x11 = self.stn(x1).cuda()
        x111 = x11.view(x1.size(0), -1).cuda()
        x = self.classifier(x111).cuda()
        return x


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


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
    count, error, correct = 0, 0, 0
    dst = labelTestDataLoader(test_dirs, imgSize)
    if use_gpu:
        testloader = DataLoader(dst, batch_size=1, shuffle=False, num_workers=8)
    else:
        testloader = DataLoader(dst, batch_size=1, shuffle=False, num_workers=0)
    start = time()
    corner = []
    for i, (XI,corner,labels, ims) in enumerate(testloader):
        count += 1
        # YI = [[int(ee) for ee in el.split('_')[:7]] for el in labels]
        if use_gpu:
            x = Variable(XI.cuda(0))
        else:
            x = Variable(XI)

        img = cv2.imread(ims[0])
        fps_pred = model(x)
        [cx, cy, w, h] = fps_pred.data.cpu().numpy()[0].tolist()
        left_up = [(cx - w/2)*img.shape[1], (cy - h/2)*img.shape[0]]
        right_down = [(cx + w/2)*img.shape[1], (cy + h/2)*img.shape[0]]

        # print("corner:",corner)
        # print("left_up,right_down:",left_up,right_down)

        prediction = (left_up[1],left_up[0],right_down[1],right_down[0])
        group_truth = (corner[0][1].item(),corner[0][0].item(),corner[1][1].item(),corner[1][0].item())
        IOU = compute_iou(prediction,group_truth)

        # print("prediction:",prediction)
        # print("group_truth:",group_truth)
        #print("IOU is:",IOU)

        if(IOU>=0.7):
            correct+=1
        else:
            error += 1

    return correct,error,count, float(correct) / count, (time() - start) / count
    #count, correct, error, float(correct) / count, (time() - start) / count


def train_model(model, optimizer, num_epochs=25):
    # since = time.time()
    best_precision = 0
    loss_log = []
    for epoch in range(epoch_start, num_epochs):
        print("epoch:",epoch)
        lossAver = []
        model.train(True)
        lrScheduler.step()
        start = time()

        for i, (XI, Y, labels, ims,total) in enumerate(trainloader):
            #test and debug
            # if(i==2):sys.exit(0)

            # print(Y,labels,ims)
            # print()
            #Y：为左上角和左下角的4个坐标值 
            #labels： 为0_0_8_9_24_30_32
            #ims：图片的路径
            # print("here1")
            if not len(XI) == batchSize:
                continue
            # print("here2")
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
            # print("here3")

            fps_pred = model(x)

            # print("here4")
            loss = 0.0

            if use_gpu:
                loss += 0.8 * nn.L1Loss().cuda()(fps_pred[:][:2], y[:][:2])#cx和cy的权重为0.8，w和h的权重为0.2
                loss += 0.2 * nn.L1Loss().cuda()(fps_pred[:][2:], y[:][2:])#这里应该有问题，这样取的每一组数据的前两维，这样写的话变成取每个batch的前两个样本了
            else:
                loss += 0.8 * nn.L1Loss()(fps_pred[:,:2], y[:,:2])#fps_pred[:][:2]->fps_pred[:,:2]
                loss += 0.2 * nn.L1Loss()(fps_pred[:,2:], y[:,2:])
            optimizer.zero_grad()#反向传播
            loss.backward()
            optimizer.step()#更新参数
            lossAver.append(loss.item())
            if i %1000  == 1:#each 50个batch写一次日志
              print ('epoch:{}[{}/{}]===>train average loss:{} spend time:{}'.format(epoch,i,torch.sum(total)//batchSize//batchSize, sum(lossAver) / len(lossAver), time()-start))
        #每个epoch之后进行evaluation
        model.eval()
        correct, error,  count, precision, avgTime = eval(model, testDirs)
        print('Evaluation epoch:{}====>precision {} avgTime {}\n'.format(epoch , precision, avgTime))

        loss_log.append(sum(lossAver) / len(lossAver))
        
        if(precision>best_precision):
            best_precision = precision
            torch.save(model.state_dict(), storeName +'-best-model')
            print("save the best model in epoch {} with {} precision".format(epoch,best_precision))

        if(epoch%5==0):#每20个epoch存一次模型
            torch.save(model.state_dict(), storeName +'stn-epoch-'+ str(epoch))#保存模型

    print("The loss information is:",loss_log)
    return model


epoch_start = int(args["start_epoch"])
resume_file = str(args["resume"])

if not resume_file == '111':#在原来的基础上继续训练模型
    # epoch_start = int(resume_file[resume_file.find('pth') + 3:]) + 1
    if not os.path.isfile(resume_file):
        print ("fail to load existed model! Existing ...")
        exit(0)
    print ("Load existed model! %s" % resume_file)
    model_conv = wR2(numClasses)
    if use_gpu:
        model_conv = torch.nn.DataParallel(model_conv, device_ids=(0,1))
        model_conv.load_state_dict(torch.load(resume_file))
        model_conv = model_conv.cuda()
    else:
        model_conv = torch.nn.DataParallel(model_conv)
        model_conv.load_state_dict(torch.load(resume_file,map_location='cpu'))

else:#从头开始训练模型
    model_conv = wR2(numClasses)
    if use_gpu:
        model_conv = torch.nn.DataParallel(model_conv, device_ids=(0,1))
        model_conv = model_conv.cuda()
    else:
        model_conv = torch.nn.DataParallel(model_conv)

print(model_conv)#打印模型结构
print(get_n_params(model_conv))#打印参数数量

# criterion = nn.CrossEntropyLoss()#采用交叉熵损失函数
# optimizer_conv = optim.RMSprop(model_conv.parameters(), lr=0.01, momentum=0.9)
optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)

#导入数据集
dst = labelFpsDataLoader(trainDirs, imgSize)#(480,480)
if use_gpu:
    trainloader = DataLoader(dst, batch_size=batchSize, shuffle=True, num_workers=8)
else:
    trainloader = DataLoader(dst, batch_size=batchSize, shuffle=True, num_workers=0)

lrScheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)
model_conv = train_model(model_conv, optimizer_conv, num_epochs=epochs)
