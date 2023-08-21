import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#from torchinfo import summary
import os
import sys





#from tensorboardX import SummaryWriter
#writer = SummaryWriter('runs/scalar_example') #可视化
#在anaconda prompt窗口中指令：tensorboard --logdir = 'runs\scalar_example' bind_all

#判断使用CPU还是GPU

transform = transforms.Compose([transforms.Resize((136,36)), #图片大小调整
                               transforms.ToTensor(),     #数据类型调整,tensor范围为0~1,
                               transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),]#归一化处理,范围为-1~1
                               )

cur_dir = sys.path[0]
test_dir = os.path.join(cur_dir, 'carIdentityData/cnn_plate_test')

plate_classes= ('有','无')

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


model = Net()
model.load_state_dict(torch.load('./车牌权重1.pth',map_location='cpu')) #测试的话，用CPU就行了
model.eval()

testset = torchvision.datasets.ImageFolder(root=test_dir , transform=transform)
testloader = DataLoader(testset , batch_size = 1 ,shuffle = False , num_workers = 0)
for i, (inputs, labels) in enumerate(testloader):
    a = inputs
    out = model(a)
    _ ,predicted = torch.max(out.data , 1)
    #print('pred=',predicted)
    print('识别结果为 ', " ".join('%5s' % plate_classes[predicted[j]] for j in range(1)))

#summary(model, (1, 3, 136, 36))
#input_to_model = torch.rand(1, 3, 20, 20)
#input_to_model= input_to_model.to(device)
#writer.add_graph(model, input_to_model)
#writer.close()