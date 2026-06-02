#-- coding:UTF-8 --

from __future__ import print_function
import glob
from itertools import chain
import os
import random
from sched import scheduler
import utils
from torch.utils.data import TensorDataset
import numpy as np
import pandas as pd
import torch
import json
import time
#import pytorch_warmup as warmup
from pathlib import Path
import scipy.io as sio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import h5py
from sklearn.metrics import cohen_kappa_score
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import joblib
import pickle
from sklearn.metrics import accuracy_score, cohen_kappa_score
from scipy.io import savemat
import os,gc
import scipy

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 256
lr = 1e-4
gamma = 0.9
seed = 42
epochs = 200
history=[]
best_acc=0

# // 模型部分

# 多分类损失函数
class MultiBranchLoss(nn.Module):
    def __init__(self, initial_weight_polar=1.0, initial_weight_temporal=1.0, initial_weight_interaction=1.0, initial_weight_vit=1.0, temporal_consistency_weight=0.1):
        super(MultiBranchLoss, self).__init__()

        self.weight_polar = nn.Parameter(torch.tensor(initial_weight_polar))
        self.weight_temporal = nn.Parameter(torch.tensor(initial_weight_temporal))
        self.weight_interaction = nn.Parameter(torch.tensor(initial_weight_interaction))
        self.weight_vit = nn.Parameter(torch.tensor(initial_weight_vit))
        self.temporal_consistency_weight = temporal_consistency_weight
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, vit_outputs, labels, polar_outputs, temporal_outputs, interaction_outputs):
        # 计算各分支损失
        labels = torch.argmax(labels, dim=1).long()
        polar_loss = self.cross_entropy_loss(polar_outputs, labels)
        temporal_loss = self.cross_entropy_loss(temporal_outputs, labels)
        interaction_loss = self.cross_entropy_loss(interaction_outputs, labels)
        vit_loss = self.cross_entropy_loss(vit_outputs, labels)

         # 计算时序一致性损失
        temporal_consistency_loss = F.mse_loss(temporal_outputs, interaction_outputs)

        # 使用权重组合损失
        total_loss = (
            self.weight_polar * polar_loss +
            self.weight_temporal * temporal_loss +
            self.weight_interaction * interaction_loss +
            self.weight_vit * vit_loss +
            self.temporal_consistency_weight * temporal_consistency_loss
        )

        return total_loss
    
class ResidualBlock3D(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock3D, self).__init__()

        self.left = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv3d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv3d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(outchannel)
            )
 
    def forward(self, x):
        out = self.left(x)
        out = F.relu(out)
        return out
    
# 时序交互模块
class TwoTimePeriodInteractionModule(nn.Module):
    def __init__(self):
        super(TwoTimePeriodInteractionModule, self).__init__()

    def forward(self, time_period1, time_period2):
        interaction_features = []

        # 遍历第一个时间段的每个特征图
        for i in range(time_period1.size(1)):
            # 对第一个时间段的每个特征图与第二个时间段的所有特征图进行相乘
            multiplied_feature = time_period1[:, i:i+1, :, :] * time_period2

            # 将结果加入列表
            interaction_features.append(multiplied_feature)

        # 将所有特征图级联在一起
        combined_feature = torch.cat(interaction_features, dim=1)

        return combined_feature

class ResidualBlockTI(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlockTI, self).__init__()

        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
 
    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class DDprTDViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, transformer, pool = 'cls', channels = 3,combine_channels):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # self.inchannel2d=32
        self.inchannel3dT=32
        self.inchannel3dP=32
        self.inchannelTI=108
       
        self.drop=nn.Dropout(0.1)
       
        self.convt1 = nn.Conv3d(6,32,(3,3,3),stride=(1,1,1),padding=(1,1,1))
        self.bn1 = nn.BatchNorm3d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.make_layer3dT(ResidualBlock3D, 64,  2, stride=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = self.make_layer3dT(ResidualBlock3D, 128,  2, stride=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.conv4 = nn.Conv3d(128,256,(3,3,3),stride=(1,1,1),padding=(1,0,0))
        self.bn4 = nn.BatchNorm3d(256)
        self.pool1 =  nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))
        self.poolT =  nn.AvgPool3d(kernel_size=(5, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.fc1=nn.Linear(256, 2048)
        self.fc2=nn.Linear(128,11)

        self.convp1 = nn.Conv3d(5,32,(3,3,3),stride=(1,1,1),padding=(1,1,1))
        self.bn1P = nn.BatchNorm3d(32)
        self.reluP = nn.ReLU(inplace=True)
        self.conv2P = self.make_layer3dP(ResidualBlock3D, 64,  2, stride=1)
        self.bn2P = nn.BatchNorm3d(64)
        self.conv3P = self.make_layer3dP(ResidualBlock3D, 128,  2, stride=1)
        self.bn3P = nn.BatchNorm3d(128)
        self.conv4P = nn.Conv3d(128,256,(3,3,3),stride=(1,1,1),padding=(1,0,0))
        self.bn4P = nn.BatchNorm3d(256)
        self.pool1P =  nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))
        self.poolP =  nn.AvgPool3d(kernel_size=(6, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.fc1P=nn.Linear(256, 2048)
        self.fc2P=nn.Linear(128,11)

        self.TIFC=TwoTimePeriodInteractionModule()
        self.convTI1 = self.make_layerTI(ResidualBlockTI, 256,  2, stride=1)
        self.bnTI1 = nn.BatchNorm2d(256)
        self.convTI2 = self.make_layerTI(ResidualBlockTI, 128,  2, stride=1)
        self.bnTI2 = nn.BatchNorm2d(128)

        num_patches = (image_size // patch_size) ** 2
        patch_dim = combine_channels * patch_size ** 2
        # print(patch_dim)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = transformer

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def make_layer3dT(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel3dT, channels, stride))
            self.inchannel3dT = channels
        return nn.Sequential(*layers)
    
    def make_layer3dP(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel3dP, channels, stride))
            self.inchannel3dP = channels
        return nn.Sequential(*layers)
    
    def make_layerTI(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannelTI, channels, stride))
            self.inchannelTI = channels
        return nn.Sequential(*layers)
    
    def forward(self, img2,imgt,imgp):

        p3d = imgp
        t3d = imgt

        out = self.convt1(t3d)
        # print("out.shape----------",out.shape)
        out = self.bn1(out)
        out = self.relu(out)      
        out = self.conv2(out)
        # print("out.shape----------",out.shape)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        # print("out.shape----------",out.shape)
        out = self.bn3(out)
        # print("out.shape----------",out.shape)
        out = self.relu(out)
        # print("out.shape----------",out.shape)
        out = self.poolT(out)
        # print("out.shape----------",out.shape)
        out= np.squeeze(out)
        # print("out.shape----------",out.shape)

        out1 = self.convp1(p3d)
        out1 = self.bn1P(out1)
        out1 = self.reluP(out1)   
        out1 = self.conv2P(out1)
        out1 = self.bn2P(out1)
        out1 = self.reluP(out1)
        out1 = self.conv3P(out1)
        out1 = self.bn3P(out1)
        out1 = self.reluP(out1)
        out1 = self.poolP(out1)
        out1= np.squeeze(out1)
        # print("out1.shape----------",out1.shape)
       
        time_period1 = img2[:,0:6,:,:]
        time_period2 = img2[:,6:12,:,:]
        time_period3 = img2[:,12:18,:,:]
        time_period4 = img2[:,18:24,:,:]

        y12 = self.TIFC(time_period1,time_period2)
        y23 = self.TIFC(time_period2,time_period3)
        y34 = self.TIFC(time_period3,time_period4)
        y2 = torch.cat([y12,y23,y34],dim=1)
        y2 = self.convTI1(y2)
        y2 = self.bnTI1(y2)
        y2 = self.relu(y2)
        y2 = self.convTI2(y2)
        y2 = self.bnTI2(y2)
        y2 = self.relu(y2)
        y2 = self.drop(y2)
        # print("y2.shape----------",y2.shape)

        # print(out.shape)  # 打印 out 的形状
        # print(out1.shape) # 打印 out1 的形状
        # print(y2.shape)   # 打印 y2 的形状

        x= torch.cat([out,out1,y2],dim=1)
        # print(x.shape)  # 打印 x 的形状
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        # print("x.shape----------",x.shape)
        x = self.to_latent(x)
        # print("x.shape----------",x.shape)

        # 在这里添加计算out, out1, y2的损失
        out_pooled = torch.mean(out, dim=(2, 3))  # 形状从[8,128,21,21]变为 [8, 128]
        out1_pooled = torch.mean(out1, dim=(2, 3))
        y2_pooled = torch.mean(y2, dim=(2, 3))
        vitloss = self.mlp_head(x)
        out = self.mlp_head(out_pooled)
        out1 = self.mlp_head(out1_pooled)
        y2 = self.mlp_head(y2_pooled)
       
        return vitloss, out, out1, y2


def trans_to_dat(matA):
    trainx=np.array(matA)
    trainx=np.nan_to_num(trainx)
    trainx=np.squeeze(trainx)
    trainx=torch.tensor(trainx).float()
    return trainx

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(seed)

efficient_transformer = Linformer(
    dim=128,
    seq_len=121+1,  # 21x21 patches + 1 cls-token
    depth=6,
    heads=8,
    k=64
)
device = 'cuda'
model = DDprTDViT(
    dim=128,
    image_size=11,
    patch_size=1,
    num_classes=16,
    transformer=efficient_transformer,
    channels=30,
    combine_channels=384,
    ).to(device)


# 初始化损失函数
loss_fn = MultiBranchLoss()


# 确保保存结果的目录存在
savepath = './result'
if not os.path.exists(savepath):
    os.makedirs(savepath)

txtfile=os.path.join(savepath,'log.txt')

# 步骤1：加载.mat文件
# 步骤1：加载.mat文件
DataPath1 = './TrainData/TrainData.mat'
DataPath2 = './ValidationData/TrainData.mat'
traindata = h5py.File(DataPath1)
valdata = h5py.File(DataPath2)


train_data = np.array(traindata['TrainData'])
train_labels = np.array(traindata['TrainLabelPro'])
test_data = np.array(valdata['TrainData'])
test_labels = np.array(valdata['TrainLabelPro'])


x_train = np.transpose(train_data,(4,0,3,1,2)) # 训练样本数据，形状为[15652, 6, 5 11, 11,]
x_train1 = np.transpose(train_data,(4,3,0,1,2)) # 训练样本数据，形状为[15652, 6, 5 11, 11,]
x_test = np.transpose(test_data,(4,0,3,1,2)) # 训练标签数据，形状为[15652, 11]
x_test1 = np.transpose(test_data,(4,3,0,1,2)) # 训练标签数据，形状为[15652, 11]
y_train = np.transpose(train_labels ,(1,0))
y_test = np.transpose(test_labels,(1,0))


trainx2D = x_train.reshape((x_train.shape[0], 30, 11, 11))
trainx3D=x_train
trainx3D1 = x_train1

trainx2D=trans_to_dat(trainx2D)
trainx3D=trans_to_dat(trainx3D)
trainx3D1=trans_to_dat(trainx3D1)
trainy=trans_to_dat(y_train)
# print(trainx2D.shape)
# print(trainx3D.shape)
# print(trainx3D1.shape)

dataset_train=TensorDataset(trainx2D,trainx3D,trainx3D1,trainy)

testx2D=x_test.reshape((x_test.shape[0], 30, 11, 11))
testx3D=x_test
testx3D1=x_test1

testx2D=np.nan_to_num(testx2D)
testx3D=np.nan_to_num(testx3D)
testx3D1=np.nan_to_num(testx3D1)

testy=np.squeeze(y_test)
testx2D=torch.tensor(testx2D)
testx2D=testx2D.float()
testx3D=torch.tensor(testx3D)
testx3D=testx3D.float() 
testx3D1=torch.tensor(testx3D1)
testx3D1=testx3D1.float() 
testy=torch.tensor(testy)
testy=testy.long()
dataset_val=TensorDataset(testx2D,testx3D,testx3D1,testy)
output_dir='./models/'
train_data_size = len(dataset_train)
valid_data_size = len(dataset_val)

train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=batch_size,
                                               drop_last=False,
                                               shuffle=True)

valid_loader = torch.utils.data.DataLoader(dataset=dataset_val,
                                              batch_size=batch_size,
                                              drop_last=False,
                                              shuffle=False)
num_steps = len(train_loader)*epochs                                             
cost = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=lr)
scheduler = CosineAnnealingLR(optimizer,T_max=num_steps)

for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs1,inputs2,inputs3,labels) in enumerate(train_loader):
            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)
            inputs3 = inputs3.to(device)
            labels = labels.to(device)
            
            #因为这里梯度是累加的，所以每次记得清零
            optimizer.zero_grad()

            vitloss,out, out1, y2 = model(inputs1,inputs2,inputs3)

            loss = cost(vitloss, torch.argmax(labels, dim=1).long())

            # outputs = model(inputs1,inputs2)
            # # 假设模型的输入是 img2, imgt 和 imgp
            # vit_outputs, polar_outputs, temporal_outputs, interaction_outputs = model(inputs1,inputs2,inputs3)

            # # 计算损失
            # loss = loss_fn(vit_outputs, labels, polar_outputs, temporal_outputs, interaction_outputs)
            # # print(loss.shape)  # 打印 loss 的形状
            # print("loss.item()",loss.item())  # 打印损失值



            loss.backward()

            optimizer.step()

            train_loss += loss.item() * inputs1.size(0)

            ret, predictions = torch.max(vitloss.data, 1)

            #correct_counts = torch.sum(predictions == torch.argmax(labels, dim=1))
            correct_counts = torch.sum(predictions == torch.argmax(labels, dim=1))
            acc = correct_counts.item() / labels.size(0)

            train_acc += acc * inputs1.size(0)

        with torch.no_grad():
            
            model.eval()

            all_labels = []
            all_predictions = []

            for j, (inputs1,inputs2,inputs3, labels) in enumerate(valid_loader):
                #print("Input shapes:", inputs1.shape, inputs2.shape)
                #print("Labels shape:", labels.shape)
                inputs1 = inputs1.to(device)
                inputs2 = inputs2.to(device)
                inputs3 = inputs3.to(device)
                labels = labels.to(device)

                # # 假设模型的输入是 img2, imgt 和 imgp
                # vit_outputs, polar_outputs, temporal_outputs, interaction_outputs = model(inputs1,inputs2,inputs3)

                # # 计算损失
                # loss = loss_fn(vit_outputs, labels, polar_outputs, temporal_outputs, interaction_outputs)

                vitloss,out, out1, y2 = model(inputs1,inputs2,inputs3)

                loss = cost(vitloss, torch.argmax(labels, dim=1).long())

                valid_loss += loss.item() * inputs1.size(0)

                ret, predictions = torch.max(vitloss.data, 1)
                # correct_counts = predictions.eq(labels.data.view_as(predictions))
                # correct_counts = torch.sum(predictions == torch.argmax(labels, dim=1))
                correct_counts = torch.sum(predictions == torch.argmax(labels, dim=1))

                acc = correct_counts.item() / labels.size(0)

                acc = torch.tensor(acc)
                valid_acc += acc.item() * inputs1.size(0)

                # 将标签和预测结果添加到列表中
                all_labels.extend(torch.argmax(labels, dim=1).cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

        scheduler.step()
        avg_train_loss = train_loss/train_data_size
        avg_train_acc = train_acc/train_data_size

        avg_valid_loss = valid_loss/valid_data_size
        avg_valid_acc = valid_acc/valid_data_size

        # 计算Kappa值
        kappa = cohen_kappa_score(all_labels, all_predictions)

        #将每一轮的损失值和准确率记录下来
        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        modelpath = os.path.join(savepath+str(epoch)+'normal.pkl')
        torch.save(model,modelpath)

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1
            modelpath = os.path.join(savepath+str(epoch)+'best.pkl')
            torch.save(model,modelpath)
        epoch_end = time.time()
        #打印每一轮的损失值和准确率，效果最佳的验证集准确率
        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
            epoch+1, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start
        ))
        print(" Kappa: {:.4f}".format(kappa))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
        with open(txtfile, "a") as myfile:
            myfile.write('epoch:'+str(epoch)+ ' ,  train_acc: '+str(avg_train_acc) + ' , train_loss: ' + str(avg_train_loss) + ' , val_acc: ' + str(avg_valid_acc) + ' , val_loss: ' + str(
                        avg_valid_loss)+', Best_Epoch: '+str(best_epoch)+' , Best Val_Acc: '+str(best_acc)+"\n")