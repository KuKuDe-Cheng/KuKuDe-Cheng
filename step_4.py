import cv2
import os
import sys
import numpy as np
from matplotlib import pyplot as plt

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

def show(img):
    cv2.imshow("demo",img)                                          #显示图片i,窗口命名为demo
    cv2.waitKey(0)                                                #显示图像时具有延时的作用,单位是毫秒按下任意键退
    cv2.destroyAllWindows() 

def pre_process(orig_img):

    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray_img', gray_img)

    blur_img = cv2.blur(gray_img, (3, 3))
    #cv2.imshow('blur', blur_img)

    sobel_img = cv2.Sobel(blur_img, cv2.CV_16S, 1, 0, ksize=3)
    sobel_img = cv2.convertScaleAbs(sobel_img)
    #cv2.imshow('sobel', sobel_img)

    hsv_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)

    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    # 黄色色调区间[26，34],蓝色色调区间:[100,124]
    blue_img = (((h > 26) & (h < 34)) | ((h > 100) & (h < 124))) & (s > 70) & (v > 70)
    blue_img = blue_img.astype('float32')

    mix_img = np.multiply(sobel_img, blue_img)
    #cv2.imshow('mix', mix_img)

    mix_img = mix_img.astype(np.uint8)

    ret, binary_img = cv2.threshold(mix_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #cv2.imshow('binary',binary_img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(21,5))
    close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow('close', close_img)

    return close_img

#判断最小矩形框是否符合车牌的 长宽比r 面积area 角度theta ，返回值为bool型
#比值在2.4~5.6  面积在400~90000   角度在正负30度
def verify_scale(rotate_rect):
   error = 0.4
   aspect = 4#4.7272
   min_area = 10*(10*aspect)#最小面积：400
   max_area = 150*(150*aspect)#最大面积：90,000
   min_aspect = aspect*(1-error)#最小比值：2.4
   max_aspect = aspect*(1+error)#最大比值：5.6
   theta = 30

   # 宽或高为0，不满足矩形直接返回False
   if rotate_rect[1][0]==0 or rotate_rect[1][1]==0:
       return False

   r = rotate_rect[1][0]/rotate_rect[1][1]
   r = max(r,1/r)
   area = rotate_rect[1][0]*rotate_rect[1][1]
   if area>min_area and area<max_area and r>min_aspect and r<max_aspect:
       # 矩形的倾斜角度在不超过theta
       if ((rotate_rect[1][0] < rotate_rect[1][1] and rotate_rect[2] >= -90 and rotate_rect[2] < -(90 - theta)) or
               (rotate_rect[1][1] < rotate_rect[1][0] and rotate_rect[2] > -theta and rotate_rect[2] <= 0)):
           return True #只有长宽比、面积、角度都满足才返回true
   return False

#泛红填充
def verify_color(rotate_rect,src_image):
    #泛洪填充算法的参数
    img_h,img_w = src_image.shape[:2]#读取原图的高、宽
    mask = np.zeros(shape=[img_h+2,img_w+2],dtype=np.uint8)#生成比原图大一点的黑色蒙板
    connectivity = 4 #种子点上下左右4邻域与种子颜色值在[loDiff,upDiff]的被涂成new_value，也可设置8邻域
    loDiff,upDiff = 30,30
    new_value = 255
    flags = connectivity
    flags |= cv2.FLOODFILL_FIXED_RANGE  #考虑当前像素与种子象素之间的差，不设置的话则和邻域像素比较
    flags |= new_value << 8
    flags |= cv2.FLOODFILL_MASK_ONLY #设置这个标识符则不会去填充改变原始图像，而是去填充掩模图像（mask）

    rand_seed_num = 5000 #生成多个随机种子
    valid_seed_num = 200 #从rand_seed_num中随机挑选valid_seed_num个有效种子
    adjust_param = 0.1
    box_points = cv2.boxPoints(rotate_rect)#找出最小区域框的四个顶点的坐标
    box_points_x = [n[0] for n in box_points]
    box_points_x.sort(reverse=False)
    adjust_x = int((box_points_x[2]-box_points_x[1])*adjust_param)
    col_range = [box_points_x[1]+adjust_x,box_points_x[2]-adjust_x]
    box_points_y = [n[1] for n in box_points]
    box_points_y.sort(reverse=False)
    adjust_y = int((box_points_y[2]-box_points_y[1])*adjust_param)
    row_range = [box_points_y[1]+adjust_y, box_points_y[2]-adjust_y]
    # 如果以上方法种子点在水平或垂直方向可移动的范围很小，则采用旋转矩阵对角线来设置随机种子点
    if (col_range[1]-col_range[0])/(box_points_x[3]-box_points_x[0])<0.4\
        or (row_range[1]-row_range[0])/(box_points_y[3]-box_points_y[0])<0.4:
        points_row = []
        points_col = []
        for i in range(2):
            pt1,pt2 = box_points[i],box_points[i+2]
            x_adjust,y_adjust = int(adjust_param*(abs(pt1[0]-pt2[0]))),int(adjust_param*(abs(pt1[1]-pt2[1])))
            if (pt1[0] <= pt2[0]):
                pt1[0], pt2[0] = pt1[0] + x_adjust, pt2[0] - x_adjust
            else:
                pt1[0], pt2[0] = pt1[0] - x_adjust, pt2[0] + x_adjust
            if (pt1[1] <= pt2[1]):
                pt1[1], pt2[1] = pt1[1] + adjust_y, pt2[1] - adjust_y
            else:
                pt1[1], pt2[1] = pt1[1] - y_adjust, pt2[1] + y_adjust
            temp_list_x = [int(x) for x in np.linspace(pt1[0],pt2[0],int(rand_seed_num /2))]
            temp_list_y = [int(y) for y in np.linspace(pt1[1],pt2[1],int(rand_seed_num /2))]
            points_col.extend(temp_list_x)
            points_row.extend(temp_list_y)
    else:
        points_row = np.random.randint(row_range[0],row_range[1],size=rand_seed_num)
        points_col = np.linspace(col_range[0],col_range[1],num=rand_seed_num).astype(np.int)

    points_row = np.array(points_row)
    points_col = np.array(points_col)
    hsv_img = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
    h,s,v = hsv_img[:,:,0],hsv_img[:,:,1],hsv_img[:,:,2]
    # 将随机生成的多个种子依次做漫水填充,理想情况是整个车牌被填充
    flood_img = src_image.copy()
    seed_cnt = 0
    for i in range(rand_seed_num):
        rand_index = np.random.choice(rand_seed_num,1,replace=False)
        row,col = points_row[rand_index],points_col[rand_index]
        # 限制随机种子必须是车牌背景色
        if (((h[row,col]>26)&(h[row,col]<34))|((h[row,col]>100)&(h[row,col]<124)))&(s[row,col]>70)&(v[row,col]>70):
            col = col[0]
            row = row[0]
            #print(col,row)
            cv2.floodFill(src_image, mask, (col,row), (255, 255, 255), (loDiff,loDiff,loDiff) , (upDiff,upDiff,upDiff) , flags)
            cv2.circle(flood_img,center=(col,row),radius=2,color=(0,0,255),thickness=2)
            seed_cnt += 1
            if seed_cnt >= valid_seed_num:
                break

    show_seed = np.random.uniform(1,100,1).astype(np.uint16)
    #cv2.imshow('floodfill'+str(show_seed),flood_img)
    #cv2.imshow('flood_mask'+str(show_seed),mask)

    # 获取掩模上被填充点的像素点，并求点集的最小外接矩形
    mask_points = []
    for row in range(1,img_h+1):
        for col in range(1,img_w+1):
            if mask[row,col] != 0:
                mask_points.append((col-1,row-1))
    mask_rotateRect = cv2.minAreaRect(np.array(mask_points))
    if verify_scale(mask_rotateRect):
        return True,mask_rotateRect
    else:
        return False,mask_rotateRect

def img_Transform(car_rect,image):
    img_h,img_w = image.shape[:2]
    rect_w,rect_h = car_rect[1][0],car_rect[1][1]
    angle = car_rect[2]
    #print('颜色处理后的最小矩形的倾斜角：',angle)

    return_flag = False
    if car_rect[2]==0:
        return_flag = True
    if car_rect[2]==90 and rect_w<rect_h:
        rect_w, rect_h = rect_h, rect_w
        return_flag = True
    if car_rect[2]==-90 and rect_w<rect_h:
        rect_w, rect_h = rect_h, rect_w
        return_flag = True

    if return_flag:
        #print('不需要矫正')
        car_img = image[int(car_rect[0][1]-rect_h/2):int(car_rect[0][1]+rect_h/2),
                  int(car_rect[0][0]-rect_w/2):int(car_rect[0][0]+rect_w/2)]
        show(car_img)
        return car_img
    
    #print('需要矫正')
    car_rect = (car_rect[0],(rect_w,rect_h),angle)
    box = cv2.boxPoints(car_rect)

    heigth_point = right_point = [0,0]
    left_point = low_point = [car_rect[0][0], car_rect[0][1]]
    for point in box:
        if left_point[0] > point[0]:
            left_point = point
        if low_point[1] > point[1]:
            low_point = point
        if heigth_point[1] < point[1]:
            heigth_point = point
        if right_point[0] < point[0]:
            right_point = point

    if left_point[1] <= right_point[1]:  # 正角度
        new_right_point = [right_point[0], heigth_point[1]]
        pts1 = np.float32([left_point, heigth_point, right_point])
        pts2 = np.float32([left_point, heigth_point, new_right_point])  # 字符只是高度需要改变
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(image, M, (round(img_w*2), round(img_h*2)))
        car_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]

    elif left_point[1] > right_point[1]:  # 负角度
        new_left_point = [left_point[0], heigth_point[1]]
        pts1 = np.float32([left_point, heigth_point, right_point])
        pts2 = np.float32([new_left_point, heigth_point, right_point])  # 字符只是高度需要改变
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(image, M, (round(img_w*2), round(img_h*2)))
        car_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
    show(car_img)
    return car_img

def locate_carPlate(orig_img,pred_image):
    #carPlate_list = []

    temp1_orig_img = orig_img.copy() #调试用
    temp2_orig_img = orig_img.copy() #调试用
    contours,heriachy = cv2.findContours(pred_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for i,contour in enumerate(contours):
        cv2.drawContours(temp1_orig_img, contours, i, (0, 255, 255), 2)
        # 获取轮廓最小外接矩形，返回值rotate_rect
        rotate_rect = cv2.minAreaRect(contour)
        # 根据矩形面积大小和长宽比判断是否是车牌
        if verify_scale(rotate_rect):
            ret,rotate_rect2 = verify_color(rotate_rect,temp2_orig_img)

            # 车牌位置矫正
            car_plate = img_Transform(rotate_rect2, temp2_orig_img)
            car_plate = cv2.resize(car_plate,(car_plate_w,car_plate_h)) #调整尺寸为后面CNN车牌识别做准备
            #carPlate_list.append(car_plate)

    return car_plate

#输入需要识别的图片的位置，输出预测结果，0表示有，1表示无
def cnn_select_carPlate(opencv_plate_dir):
    model = Net()
    model.load_state_dict(torch.load('./车牌权重1.pth',map_location='cpu')) #测试的话，用CPU就行了
    model.eval()
    plateset = torchvision.datasets.ImageFolder(root=opencv_plate_dir , transform=transform)
    plateloader = DataLoader(plateset , batch_size = 1 ,shuffle = False , num_workers = 0)
    for i, (inputs, labels) in enumerate(plateloader):
        a = inputs
        out = model(a)
        _ ,predicted = torch.max(out.data , 1)
        predicted = predicted.item()
        
        #print('识别结果为：', plate_classes[predicted])
        return predicted

img = cv2.imread("carIdentityData/pictures/1.jpg",cv2.IMREAD_UNCHANGED)#读图
pred_img = pre_process(img)#预处理

car_plate_w,car_plate_h = 136,36 #定义车牌大小
car_plate = locate_carPlate(img,pred_img) #车牌定位


cv2.imwrite("carIdentityData/cnn_plate_test/test/1.jpg",car_plate)
cur_dir = sys.path[0]
opencv_plate_dir = os.path.join(cur_dir, 'carIdentityData/cnn_plate_test')
plate_classes= ('有','无')
transform = transforms.Compose([transforms.Resize((136,36)), #图片大小调整
                               transforms.ToTensor(),     #数据类型调整,tensor范围为0~1,
                               transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),]#归一化处理,范围为-1~1
                               )

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




predicted = cnn_select_carPlate(opencv_plate_dir)

#开始字符分割
## 根据每行和每列的黑色和白色像素数进行图片分割。

# 1、读取图像，并把图像转换为灰度图像并显示
img_ = car_plate  # 读取图片
img_gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)  # 转换了灰度化
# cv2.imshow('gray', img_gray)  # 显示图片
# cv2.waitKey(0)

# 2、将灰度图像二值化，设定阈值是100
ret, img_thre = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
show(img_thre)

# 4、分割字符
white = []  # 记录每 一列的白色像素总和
black = []  # ..........黑色.......
height = img_thre.shape[0]
width = img_thre.shape[1]
white_max = 0
black_max = 0
# 计算每一列的黑白色像素总和
for i in range(width):
    s = 0  # 这一列白色总数
    t = 0  # 这一列黑色总数
    for j in range(height):
        if img_thre[j][i] == 255:
            s += 1
        if img_thre[j][i] == 0:
            t += 1
    white_max = max(white_max, s)
    black_max = max(black_max, t)
    white.append(s)
    black.append(t)
    #print(s)
    #print(t)

# 分割图像
def find_end(start_):
    end_ = start_ + 1
    for m in range(start_ + 1, width  ):
        if (black[m] ) > (0.95 * black_max ):  # 0.95这个参数请多调整，对应下面的0.05（针对像素分布调节）
            end_ = m
            break
        if m == width :
            end_ = m
    return end_

n = 0
start = 0
end = 1
word = []
char_num = 0
while n < width-1:
    n += 1
    if (white[n] ) > (0.05 * white_max ):
        # 0.05这个参数请多调整，对应上面的0.95
        start = n
        end = find_end(start)
        n = end
        if end - start > 5:
            if end == width-1:
                cj = img_thre[1:height, start:end]
            else:
                cj = img_thre[1:height, start:end+1]
            cj = cv2.resize(cj, (15, 30))
            char_num = char_num +1
            cv2.imwrite("carIdentityData/cnn_char_test/identify/%d.jpg"%char_num,cj)
            #show(cj)
            word.append(cj)

print('找出的字符数量：',len(word))

for i,j in enumerate(word):
    plt.subplot(1,8,i+1)
    plt.imshow(word[i],cmap='gray')
plt.show()
 
