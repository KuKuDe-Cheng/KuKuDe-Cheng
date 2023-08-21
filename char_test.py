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



transform = transforms.Compose([transforms.Resize((20,20)), #图片大小调整
                               transforms.ToTensor(),     #数据类型调整,tensor范围为0~1,
                               transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),]#归一化处理,范围为-1~1
                               )

cur_dir = sys.path[0]

test_dir = os.path.join(cur_dir, 'carIdentityData/cnn_char_test')
#test_dir = os.path.join(cur_dir, 'carIdentityData/opencv-char')




char_classes= ('0','1','2','3','4','5','6','7','8','9','A','B',
           'C','D','E','F','G','H','J','K','L','M','N','P',
           'Q','R','S','T','U','V','W','X','Y','Z',
           '川','鄂', '赣', '甘', '贵' ,'桂' ,'黑' ,'沪','冀','津' ,'京','吉','辽','鲁','蒙',
           '闽','宁','青','琼','陕','苏','晋','皖','湘','新','豫','渝','粤','云','藏','浙')

class Net(nn.Module):  
    def __init__(self):
        super(Net ,self).__init__() 
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3 , out_channels=16 , kernel_size=3, stride=1,padding=1),  
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=2),  
        )
        self.conv2 = nn.Sequential(  
            nn.Conv2d(in_channels=16 , out_channels=32 , kernel_size=3, stride=1,padding=1),  
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=2),  
        )
        self.conv3 = nn.Sequential(  
            nn.Conv2d(in_channels=32 , out_channels=16 , kernel_size=3, stride=1,padding=1),  
            nn.ReLU(),  
        )
        self.fc1 = nn.Linear(16*5*5 , 300)
        self.fc2 = nn.Linear(300 , 150)
        self.fc3 = nn.Linear(150 , 130)
        self.fc4 = nn.Linear(130 , 65)       

    def forward(self , x):  
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
       
        x = x.view( -1 ,16*5*5)  
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = x.view(x.size(0), -1)
        return x




model = Net()

model.load_state_dict(torch.load('./字符权重7.pth',map_location='cpu')) #测试的话，用CPU就行了
model.eval()


testset = torchvision.datasets.ImageFolder(root=test_dir , transform=transform)
testloader = DataLoader(testset , batch_size = 1 ,shuffle = False , num_workers = 0)
for i, (inputs, labels) in enumerate(testloader):
    a = inputs
   
    out = model(a)
    _ ,predicted = torch.max(out.data , 1)
    print('识别结果为 ', " ".join('%5s' % char_classes[predicted[j]] for j in range(1)))

#summary(model, (1, 3, 20, 20))

#input_to_model = torch.rand(1, 3, 20, 20)
#input_to_model= input_to_model.to(device)
#writer.add_graph(model, input_to_model)
#writer.close()
