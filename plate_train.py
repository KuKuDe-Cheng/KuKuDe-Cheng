import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


import numpy as np
import os
import sys
import time


from tensorboardX import SummaryWriter
writer = SummaryWriter('runs')  #可视化



from PIL import Image  
import PIL.ImageOps
import matplotlib.pyplot as plt

#判断使用CPU还是GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
learning_rate=0.002

transform = transforms.Compose([transforms.Resize((136,36)), #图片大小调整,这里是双括号！！！
                               transforms.ToTensor(),     #数据类型调整
                               transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]#归一化处理
                               )

cur_dir = sys.path[0]
data_dir = os.path.join(cur_dir, 'carIdentityData/cnn_plate_train')
test_dir = os.path.join(cur_dir, 'carIdentityData/cnn_plate_test')

trainset = torchvision.datasets.ImageFolder(root=data_dir , transform=transform)
trainloader = DataLoader(trainset , batch_size = 4 ,shuffle = True , num_workers = 0)
#数据是经过归一化处理的tesnor。标签是文件夹的序号，如0、1

class Net(nn.Module):  
    def __init__(self):
        super(Net ,self).__init__() 
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3 , out_channels=16 , kernel_size=5, stride=1,padding=2),  
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=2),  
        )
        self.conv2 = nn.Sequential(  
            nn.Conv2d(in_channels=16 , out_channels=32 , kernel_size=5, stride=1,padding=2),  
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=2),  
        )
        self.conv3 = nn.Sequential(  
            nn.Conv2d(in_channels=32 , out_channels=16 , kernel_size=5, stride=1,padding=2),  
            nn.ReLU(),  
        )
        self.fc1 = nn.Linear(16*34*9 , 500)
        self.fc2 = nn.Linear(500 , 250)
        self.fc3 = nn.Linear(250 , 120)
        self.fc4 = nn.Linear(120 , 2)       

    def forward(self , x):  
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
       
        x = x.view( -1 ,16*34*9)  
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = x.view(x.size(0), -1)
        return x

model = Net().to(device)
#print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
total_step = len(trainset)
print('训练集的长度：',total_step)

time_start = time.time()
for epoch in range(epochs):
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(trainloader):
        
        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        

        if i % 50 == 0:
            print('[epoch=%d , %5d]当前组平均loss:%.3f' % (epoch + 1, i + 1, running_loss / 200))
            writer.add_scalars('loss',{"train":running_loss/200}) #用于记录数据画图
            running_loss = 0.0
          

              
        
            
time_end = time.time() - time_start
print("在GPU上运行训练网络所消耗的时间(s):", time_end)

print('完成训练!!!!!!!!!!!!!')
torch.save(model.state_dict(), '车牌权重1.pth')  