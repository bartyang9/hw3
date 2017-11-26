#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import torch #
import torchvision
import torchvision.datasets as dset #
import torchvision.transforms as transforms #
from torch.utils.data import DataLoader, Dataset #
import matplotlib.pyplot as plt
import torchvision.utils #
import numpy as np
import random
from PIL import Image
from torch.autograd import Variable #
import PIL.ImageOps
import torch.nn as nn #
from torch import optim
import torch.nn.functional as F
import os 
import re

'''create the image folder list'''
def reader(r,mode):
    path = ''.join([r,mode])
    file = open(path)
    result = []
    for line in file:
        objects = line.split()
        result.append(objects)
    return result


'''Custom Dataset'''
class lfwDataset(Dataset):
    def __init__(self,root,reader,augment,transform=None):
        self.root = root
        self.augment = augment
        self.transform = transform
        self.reader = reader
        
    def __getitem__(self,index):
        root_dir = self.root
        image_tuple = self.reader[index]
        img1_dir = ''.join([root_dir,image_tuple[0]])
        img2_dir = ''.join([root_dir,image_tuple[1]])
        img1 = Image.open(img1_dir)
        img2 = Image.open(img2_dir)
        label = image_tuple[2]
        if self.transform is not None:
            if self.augment is True:
                img1 = self.augmentation(img1)
                img2 = self.augmendation(img2)
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2, label
    
    def augmendation(self, img0):
        rotate_range = random.uniform(-30,30)
        translation_range = random.uniform(-10,10)
        scale_range = random.uniform(0.7,1,3)
        if np.random.random() < 0.7:
            img0 = img0.rotate(rotate_range)
        if np.random.random() < 0.7:
            img0 = img0.transform((img0.size[0],img0.size[1]), Image.AFFINE, (1,0,translation_range,0,1,translation_range))
        if np.random.random() < 0.7:
            img0 = img0.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.random() < 0.7:
            img0 = img0.resize((int(128*scale_range),int(128*scale_range)))
            width, height = img0.size
            half_the_width = width / 2
            half_the_height = height / 2
            img0 = img0.crop((half_the_width - 64, half_the_height - 64,
                              half_the_width + 64, half_the_height + 64))
        return img0
    
    def __len__(self):
        return len(self.lst)
    
# =============================================================================
# class SiameseNetWork(nn.Module):
#     def __init__(self):
#         super(SiameseNetWork,self).__init__()
#         self.cnn = nn.Sequential(
#                 nn.Conv2d(3, 64, kernel_size=5, padding=2),      
#                 nn.ReLU(inplace=True),
#                 nn.BatchNorm2d(64),                              
#                 nn.MaxPool2d(2,stride = 2),                              
#                 
#                 nn.Conv2d(64, 128, kernel_size=5, padding=2),      
#                 nn.ReLU(inplace=True),
#                 nn.BatchNorm2d(128),                              
#                 nn.MaxPool2d(2,stride = 2),                             
#                 
#                 nn.Conv2d(128, 256, kernel_size=3, padding=2),      
#                 nn.ReLU(inplace=True),
#                 nn.BatchNorm2d(256),                              
#                 nn.MaxPool2d(2, stride=2),                             
#                 
#                 nn.Conv2d(256, 512, kernel_size=3, padding=1),      
#                 nn.ReLU(inplace=True),
#                 nn.BatchNorm2d(512),
#                 )
#         
#         self.fc = nn.Sequential(
#                 nn.Linear(16*16*512,1024),
#                 nn.ReLU(inplace=True),
#                 nn.BatchNorm1d(1024)
#                 )
#         self.fcc = nn.Sequential(nn.Linear(2048,1))
#     
#     def forward_once(self,x):
#         x = self.cnn(x)
#         x = x.view(x.size()[0], -1)
#         x = self.fc(x)
#         return x
#     
#     def forward(self,input1, input2):
#         output1 = self.foward_once(input1)
#         output2 = self.foward_once(input2)
#         output = torch.cat((output1,output2),1)
#         output = self.fcc(output)
#         output = torch.sigmoid(output)
#         return output
# =============================================================================
    
    
"building CNN"
class SiameseNetWork (nn.Module):
    def __init__(self):
        super(SiameseNetWork, self).__init__()
        self.Cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=2),     # 1
            nn.ReLU(inplace=True),                                                                       # 2
            nn.BatchNorm2d(num_features=64),                                                             # 3
            nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2)),                                              # 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5,5), stride=(1, 1), padding=2),    # 5
            nn.ReLU(inplace=True),                                                                       # 6
            nn.BatchNorm2d(num_features=128),                                                            # 7
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),                                             # 8
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),  # 9
            nn.ReLU(inplace=True),                                                                       # 10
            nn.BatchNorm2d(num_features=256),                                                            # 11
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),                                             # 12
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),  # 13
            nn.ReLU(inplace=True),                                                                       # 14
            nn.BatchNorm2d(num_features=512)                                                             # 15
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=131072, out_features=1024),                                            # 17
            nn.ReLU(inplace=True),                                                                       # 18
            nn.BatchNorm2d(num_features=1024)                                                            # 19
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1),
            
        )
    def forward_once(self, x):
        output = self.Cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output
    def forward(self, input1, input2):
        input1 = self.forward_once(input1)
        input2 = self.forward_once(input2)
        output = torch.cat((input1, input2), 1)
        output = self.fc1(output)
        output = torch.sigmoid(output)
        return output


class Config():
    training_dir =  '/home/yikuangy/hw3/lfw/' 
    batch_size = 64
    train_epochs = 100
    split_dir = '/home/yikuangy/hw3/'
    
'''overload the plotting function'''
def imshow(img):
    plt.axis("off")
    img_np = img.numpy()
    plt.imshow(np.transpose(img_np,(1,2,0)))
    plt.show()
    
'''create traning and testing reader&dataset'''
readersTrain = reader(Config.split_dir, 'train.txt')
lfw_train = lfwDataset(root=Config.training_dir,augment=True,reader=readersTrain,
                       transform=transforms.Compose([transforms.Scale((128,128)),transforms.ToTensor()]))
readerTest = reader(Config.split_dir,'test.txt')
lfw_test = lfwDataset(root=Config.training_dir,augment=False,reader=readerTest,
                      transform=transforms.Compose([transforms.Scale((128,128)),transforms.ToTensor()]))
data_train = DataLoader(lfw_train, batch_size=Config.batch_size,shuffle=True,num_workers=8)
data_test = DataLoader(lfw_test, batch_size=Config.batch_size,shuffle=False,num_workers=8)

'''train'''
net = SiameseNetWork().cuda
loss = nn.BCELoss()
optimiz = optim.Adam(params=net.parameters(),lr = 0.00001)
count = []
loss_log = []
iter_num = 0
for epoch in range(Config.train_epochs):
    for i,data in enumerate(data_train,0):
        img1,img2,label = data
        label = label.type(torch.FloatTensor).cuda()
        img1,img2,label = Variable(img1).cuda(), Variable(img2).cude(), Variable(label).cuda()
        output = net(img1,img2)
        optimiz.zero_grad()
        loss_BCE = loss(output,label)
        loss_BCE.backward()
        optimiz.step()
        if i % 10 == 0:
            print("Epoch num {}\n Current loss {}\n".format(epoch, loss_BCE.data[0]))
            iter_num += 10
            count.append(iter_num)
            loss_log.append(loss.data[0])
            
torch.save(net.state_dict(),f='p1a_model')

net.load_state_dict(torch.load(f='p1a_model'))

'''train testing'''
total = 0
correct = 0
for i,data_test in enumerate(data_train,0):
    img1Test,img2Test,labelTest = data_test
    img1Test,img2Test,labelTest = Variable(img1Test,volatile=True).cuda(), Variable(img2Test,volatile=True).cude(), Variable(labelTest).cuda()
    labelTest = labelTest.type(torch.FloatTensor).cuda()
    labelTest = labelTest.type('torch.LongTensor')
    output = net.foward(img1Test,img2Test)
    output = (torch.round(output)).type('torch.LongTensor')
    total += labelTest.size(0)
    correct += (output == labelTest).sum().type('torch.LongTensor')
correct = correct.data.numpy().astype(np.float)
accuracy = (100*correct/total)
print('Accuracy of the network on trained images: %d %%' % accuracy)

'''train testing'''
total = 0
correct = 0
for i,data_test in enumerate(data_test,0):
    img1Test,img2Test,labelTest = data_test
    img1Test,img2Test,labelTest = Variable(img1Test,volatile=True).cuda(), Variable(img2Test,volatile=True).cude(), Variable(labelTest).cuda()
    labelTest = labelTest.type(torch.FloatTensor).cuda()
    labelTest = labelTest.type('torch.LongTensor')
    output = net.foward(img1Test,img2Test)
    output = (torch.round(output)).type('torch.LongTensor')
    total += labelTest.size(0)
    correct += (output == labelTest).sum().type('torch.LongTensor')
correct = correct.data.numpy().astype(np.float)
accuracy = (100*correct/total)
print('Accuracy of the network on trained images: %d %%' % accuracy)