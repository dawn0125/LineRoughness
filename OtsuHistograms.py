# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:40:53 2023

@author: D.Zheng
"""
import cv2 as cv 
import os 
import numpy as np
import matplotlib.pyplot as plt

# main
image_folder = '//wp-oft-nas/HiWis/GM_Dawn_Zheng/Arvid/Magnesium Walls for Dawn/otsutest'
loi = os.listdir(image_folder)
acceptedFileTypes = ['png'] # add more as needed

for i in loi:
    if( '.' in i and i.split('.')[-1] in acceptedFileTypes):
        f = image_folder + '/' + i
        img = cv.imread(f)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
        # plot
        plt.subplot(121), plt.imshow(img)
        plt.title(i), plt.xticks([]), plt.yticks([])

        plt.subplot(122), plt.imshow(thresh)
        plt.title('Otsu Threshing'), plt.xticks([]), plt.yticks([])
        
        plt.tight_layout()
        plt.show()
        
        plt.hist(img.ravel(),256)
        plt.title(i + ' Histogram'), plt.yticks([])
        plt.grid()
        plt.show()