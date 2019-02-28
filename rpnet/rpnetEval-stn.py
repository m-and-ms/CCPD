#encoding:utf-8
import cv2
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import argparse
import numpy as np
from os import path, mkdir
from load_data import *
from time import time
from roi_pooling import roi_pooling_ims
from shutil import copyfile

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",required=True,
                help="path to the input folder")#default="C:/Users/Jet Zhang/Desktop/PR_corner/ccpd_final/home/booy/ccpd_dataset/ccpd_fn/"
ap.add_argument("-m", "--model", default="./model/fh02.pthstn-best-model",
                help="path to the model file")#required=True

args = vars(ap.parse_args())
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
use_gpu = torch.cuda.is_available()
print (use_gpu)

numClasses = 4
numPoints = 4
imgSize = (480, 480)
batchSize = 1 if use_gpu else 1
resume_file = str(args["model"])

provNum, alphaNum, adNum = 38, 25, 35
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

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

        self.localization = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=5),#[4, 512, 7, 7]
            # nn.MaxPool2d(2),
            nn.ReLU(True),

            nn.Conv2d(192, 128, kernel_size=3),#torch.Size([4, 512, 7, 7])
            # nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )            
         # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(3200, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)#torch.Size([4, 512, 5, 5])
        xs = xs.view(xs.size(0), -1)#xs.view(-1, 512*7*7)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        # #
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        x1 = self.features(x)
        x11 = self.stn(x1)
        x111 = x11.view(x1.size(0), -1)
        x = self.classifier(x111)
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

        x9 = self.wR2.module.stn(x9)
        # x91 = self.localization(x9)
        # x91 = x91.view(x91.size(0), -1)
        # theta = self.fc_loc(x91)
        # theta = theta.view(-1, 2, 3)
        # grid = F.affine_grid(theta, x9.size())
        # x9 = F.grid_sample(x9, grid)
        x9 = x9.view(x9.size(0),-1)
        # print("x9.shape:",x9.shape)

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
        # print("boxNew:",boxNew)
        # print("boxNew.mm(p1):",boxNew.mm(p1))
        # print("_x1.shape:",_x1.shape)
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


def isEqual(labelGT, labelP):
    # print (labelGT)
    # print (labelP)
    compare = [1 if int(labelGT[i]) == int(labelP[i]) else 0 for i in range(7)]
    # print(sum(compare))
    return sum(compare)


model_conv = fh02(numPoints, numClasses)
if use_gpu:
    model_conv = torch.nn.DataParallel(model_conv, device_ids=(0,1))
    model_conv.load_state_dict(torch.load(resume_file))
    model_conv = model_conv.cuda()
else:
    model_conv = torch.nn.DataParallel(model_conv)
    model_conv.load_state_dict(torch.load(resume_file,map_location='cpu'))

model_conv.eval()

count, detectionCorrect, recognitionCorrect = 0, 0, 0

dst = labelTestDataLoader(args["input"].split(','), imgSize)
if use_gpu:
    testloader = DataLoader(dst, batch_size=1, shuffle=False, num_workers=8)
else:
    testloader = DataLoader(dst, batch_size=1, shuffle=False, num_workers=0)

with open('fh0Eval', 'wb') as outF:
    pass

start = time()
status = False
    
for i, (XI,corner,labels, ims) in enumerate(testloader):
    count += 1
    #测试识别准确率
    YI = [[int(ee) for ee in el.split('_')[:7]] for el in labels]
    if use_gpu:
        x = Variable(XI.cuda(0))
    else:
        x = Variable(XI)
    # Forward pass: Compute predicted y by passing x to the model

    fps_pred, y_pred = model_conv(x)

    outputY = [el.data.cpu().numpy().tolist() for el in y_pred]
    labelPred = [t[0].index(max(t[0])) for t in outputY]

    # compare YI, outputY
    try:
        if isEqual(labelPred, YI[0]) == 7:
            recognitionCorrect += 1
            status = True
        else:
            status = False
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
    print("iou is:",IOU, "Rec pre:",labelPred," Rec label:",YI[0]," status:",status)
    if(IOU>=0.5):
        detectionCorrect+=1
    #test code block
    if i%100==0:
        print("The detection AP:{} classification ap is:{}".format(detectionCorrect/count, recognitionCorrect/count))

print("The final evaluation result is: total number of example:{} detection accuracy:{} classfication accuracy:{} each example spend time:{}".format(count,detectionCorrect/count, recognitionCorrect/count, (time() - start) / count))
