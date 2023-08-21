#求导后与蓝色区域叠加
import cv2
import os
import sys
import numpy as np

def show(img):
    cv2.imshow("demo",img)                                          #显示图片i,窗口命名为demo
    cv2.waitKey(0)                                                #显示图像时具有延时的作用,单位是毫秒按下任意键退
    cv2.destroyAllWindows()  

img = cv2.imread("carIdentityData/pictures/1.jpg",cv2.IMREAD_UNCHANGED) 
#show(img)
gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#show(gray_img)

blur_img = cv2.blur(gray_img, (3, 3))
#show(blur_img)

sobel_img = cv2.Sobel(src=blur_img, ddepth=cv2.CV_16S, dx=1, dy=0, ksize=3)
sobel_img = cv2.convertScaleAbs(sobel_img)
#show(sobel_img)

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
blue_img = (((h > 26) & (h < 34)) | ((h > 100) & (h < 124))) & (s > 70) & (v > 50)
blue_img = blue_img.astype('float32')
#show(blue_img)

mix_img = np.multiply(sobel_img, blue_img)
#show(mix_img)

mix_img = mix_img.astype(np.uint8)
#show(mix_img)

ret, binary_img = cv2.threshold(mix_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#show(binary_img)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(21,5))
#print(kernel)
close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Original image',img)
cv2.imshow('Image processing 1',close_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

