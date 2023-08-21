import cv2
import os
import sys
import numpy as np

def show(img):
    cv2.imshow("demo",img)                                          #显示图片i,窗口命名为demo
    cv2.waitKey(0)                                                #显示图像时具有延时的作用,单位是毫秒按下任意键退
    cv2.destroyAllWindows() 

img = cv2.imread("carIdentityData/pictures/1.jpg",cv2.IMREAD_UNCHANGED) 
car_plate_w,car_plate_h = 136,36

#图片处理1
gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur_img = cv2.blur(gray_img, (3, 3))
sobel_img = cv2.Sobel(src=blur_img, ddepth=cv2.CV_16S, dx=1, dy=0, ksize=3)
sobel_img = cv2.convertScaleAbs(sobel_img)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
blue_img = (((h > 26) & (h < 34)) | ((h > 100) & (h < 124))) & (s > 70) & (v > 70)
blue_img = blue_img.astype('float32')
mix_img = np.multiply(sobel_img, blue_img)
mix_img = mix_img.astype(np.uint8)
ret, binary_img = cv2.threshold(mix_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(21,5))
close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)


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
    cv2.imshow('floodfill'+str(show_seed),flood_img)
    cv2.imshow('flood_mask'+str(show_seed),mask)

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
    print('颜色处理后的最小矩形的倾斜角：',angle)

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
        print('不需要矫正')
        car_img = image[int(car_rect[0][1]-rect_h/2):int(car_rect[0][1]+rect_h/2),
                  int(car_rect[0][0]-rect_w/2):int(car_rect[0][0]+rect_w/2)]
        show(car_img)
        return car_img
    
    print('需要矫正')
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

#图片处理2
carPlate_list = []
temp1_orig_img = img.copy()
temp2_orig_img = img.copy()
contours,heriachy = cv2.findContours(close_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#最外层的框框，而且只保留框框轮廓角点的坐标
for i,contour in enumerate(contours):
    cv2.drawContours(temp1_orig_img, contours, i, (0,255, 255), 2)
    #show(temp1_orig_img)
    rotate_rect = cv2.minAreaRect(contour)
    
    if verify_scale(rotate_rect):
        print('i=',i)
        ret,rotate_rect2 = verify_color(rotate_rect,temp2_orig_img)
        #print(ret)
        
        car_plate = img_Transform(rotate_rect2, temp2_orig_img)
        car_plate = cv2.resize(car_plate,(car_plate_w,car_plate_h))
        
        carPlate_list.append(car_plate)

#print(carPlate_list)


