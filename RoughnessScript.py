# -*- coding: utf-8 -*-
"""
This script accepts a folder of images, each containing multiple sample. 
3 samples will be extracted from each image based on size. From each contour, both
sides will be fitted via linear regression. The mean, absolute deviation from the fit 
is recorded as the roughness. 

How to use:
    1. Change sourcePath to the directory where all your folders will be saved 
    2. Change the inDir to the folder of your images 
    3. Set scale if it is different from current number 
    
Notes:
    1. samples are oriented vertically 
    2. samples are supposed to be the height of the image (can change this in the checkContour fnc)
    3. 28/30 effectiveness... should check the summary images to validate 
    
Procedure: 
    1. From each image, first filter via morphological closing 
    2. Select the 3 largest contours (excluding the thick strip on the right and the merged part on the left)
    3. Check that 3 largest contours are the height of the image
        3a. if not, filter without morphological closing and repeat step 2
    4. Save ROIs
    5. From each ROI, find edges and remove pores (according to area)
    6. Separate image between left and right
    7. To each side, fit a linear regression using SciPy 
    8. Find mean, absolute deviation between actual points and fitted points 
    9. Record roughness 
    10. Repeat for entire folder of images 
    11. Export to excel 
"""

#===========================IMPORT STATEMENTS==================================
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy import ndimage
import scipy
import os
import pandas as pd

#===============================FUNCTIONS======================================
def threshManual(img, lower, upper):
    '''
    thresh according to bins added manually 
    '''
    img = ndimage.gaussian_filter(img, 2, mode='nearest')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, lower, upper, cv.THRESH_BINARY)
    return thresh
    

def threshOtsu(img):
    '''
    img: image array 
    thresh: black and white image array 

    '''
    img = ndimage.gaussian_filter(img, 2, mode='nearest')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return thresh

def morph(img):
    '''
    perform morphological operations to remove noise
    '''
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(6,6))
    closed = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    
    return closed 

def findContour(img):
    if np.ndim(img) != 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    return contours 

def findAreas(contours):
    '''
    contours: image array with contours found 
    area: array with area of each contour 

    '''
    area = []
    for cnt in contours:
        area.append(cv.contourArea(cnt))
    return np.asarray(area)

def cntsOI(cnts, lower, upper):
    '''
    return indices of contours of interest 
    '''
    areas = findAreas(cnts)
    smalltobig = np.argsort(areas)
    
    i = smalltobig[lower:upper]
    
    return i

def extractROI(contour):
    """
    contours: image array with contours found 
    ROI: image array with region of interest
    """
    x,y,w,h = cv.boundingRect(contour) # draws a straight rectange around the contour 
    ROI = img[y:y+h, x:x+w] # crops image with array indexing 
    return ROI

def findHeights(cnts, cntsOI):
    '''
    cnts: image array of contours 
    cntsOI: indices of contours of interest 
    contour_heights: list of heights of the contours 

    '''
    cnt_heights = []
    for i in cntsOI:
        x,y,w,h = cv.boundingRect(cnts[i]) 
        cnt_heights.append(h) # add heights to cnt heights 
    return cnt_heights

def filterHeight(cnts, ratio, height):
    '''
    return list of tall contours 
    '''
    tall_cnts = []
    for cnt in cnts: 
        x,y,w,h = cv.boundingRect(cnt) 
        if h >= ratio * height:
            tall_cnts.append(cnt)
    return tall_cnts   


def orderMass(cnts):
    '''
    returns indices of contours sorted from left to right, based on center of mass 
    '''
    com = []
    for cnt in cnts:
        M = cv.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        com.append(cx)
    return np.argsort(com) 

def filterPores(img): # from stack overflow, works with all the images
    '''
    img: original image being analyzed 
    final_edges: image of the edges extracted from img 
    gets rid of the pores
    '''
    blurred = ndimage.gaussian_filter(img, 2, mode='nearest')
    edges = cv.Canny(blurred,127,175)

    # Apply morphological operation (closing) to the EDGES so they are connected
    kernel = np.ones((5,3),np.uint8)
    edges_closing = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    nb_blobs, im_with_separated_blobs, stats, _ = cv.connectedComponentsWithStats(edges_closing)
    sizes = stats[:, -1]
    sizes = sizes[1:]
    nb_blobs -= 1
    min_size = 80  

    final_edges = np.zeros_like(img)
    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:
            final_edges[im_with_separated_blobs == blob + 1] = 255
            
    return final_edges

def extractCoords(edges):
    '''
    edges: image of edges
    x: x coordinate(s)
    y: y coordinates(s)
    '''
    y, x = np.where(edges > 0)
    return x, y

def splitEdges(edges):
    """
    return 2 image arrays, one only of left edge and one only of right edge
    """
    midpt = np.shape(edges)[1] // 2 
    
    left = edges.copy()
    left[:, midpt:] = 0
    
    right = edges.copy()
    right[:, :midpt] = 0 
    
    return left, right 

def fit(x, y):
    '''
    fits according to linear regression 
    x, y: coordinates being fitted 
    xfit: linear regression array 

    '''
    slope, intercept, r, p, se = scipy.stats.linregress(y, x)
    xfit = slope * y + intercept
    return xfit

def avgDev(actual, fit):
    return np.mean(np.abs(actual - fit))   

#=============================MAIN========================================
scale = 5.88 
sourcePath = '//wp-oft-nas/HiWis/GM_Dawn_Zheng/Arvid/Magnesium Walls for Dawn/'
inDir = sourcePath + 'Post Processed' 
loi = []
data = []
acceptedFileTypes = ["png"] # add more as needed
saveSummaries = False
height_scale = 0.9 # ratio of the image height to accept contours 

# threshing method, (127, 255) is a standard place to start
manual_threshing = False
thresh_upper_bound = 255
thresh_lower_bound = 127 

# indices of contours to be extracted based on size 
cnt_lower_bound = -4
cnt_upper_bound = -1

# column names in excel file are based on the number of ROI specified by indices above
column_names = np.arange(np.abs(cnt_lower_bound - cnt_upper_bound))

# for every picture in your directory, to extract the contours for analysis 
for i in os.listdir(inDir): 
    if( '.' in i and i.split('.')[-1] in acceptedFileTypes):
        f = inDir + '/' + i
        loi.append(i)
        print('Processing ' + i)
        
        # make output directory if it doesn't exist already 
        outDir = sourcePath + ' ROI_Images/' + i + '_output'
        if not os.path.exists(outDir):
            os.makedirs(outDir)
        
        # read img
        img = cv.imread(f) 
        
        # extract img height for morph checker
        img_height = np.shape(img)[0] 
        
        # copy img for summary img
        summary_image = img.copy() 
    
        # different threshing methods 
        if manual_threshing == True:
            thresh = threshManual(img, thresh_lower_bound, thresh_upper_bound)
        elif manual_threshing == False:
            thresh = threshOtsu(img)
        
        # try to morph first (19/30 success rate)
        morph_img = morph(thresh)
        cnts = findContour(morph_img) 
        cnts = filterHeight(cnts, height_scale, img_height)
        order = orderMass(cnts)
        indices = order[cnt_lower_bound : cnt_upper_bound]
        
        
        # extracting contours
        ROI_number = 0
        for j in indices:
            x,y,w,h = cv.boundingRect(cnts[j])
            ROI = img[y:y+h, x:x+w]
            cv.imwrite('{}/{} ROI{}'.format(outDir, i, ROI_number) + '.png', ROI)
            cv.rectangle(summary_image,(x,y),(x+w,y+h),(255,255,0),2) # also add label later on
            ROI_number += 1
       
        
        # make summary directory if it doesn't exist 
        sumimgDir = sourcePath + '/summary_imgs'
        if not os.path.exists(sumimgDir):
            os.makedirs(sumimgDir)
        
        
        #show and save extraction summary image
        if saveSummaries == True:
            cv.imwrite(sumimgDir + '/' + i, summary_image)
        plt.imshow(summary_image)
        plt.title(i + ' ROI')
        plt.show()
        
        
        # begin roughness script   
        allRoughness = []
        for ROI in os.listdir(outDir):
            if( '.' in ROI and ROI.split('.')[-1] in acceptedFileTypes):
                try: 
                    img = cv.imread(os.path.join(outDir, ROI), cv.IMREAD_GRAYSCALE) 
                    roughness_img = cv.imread(os.path.join(outDir, ROI))
                    edges = filterPores(img)
                    roughness = []
                    
                    # treat each side separately (should loop 2x)
                    for side in splitEdges(edges):
                        # extract coordinates and fit
                        x, y = extractCoords(side)
                        xfit = fit(x, y)
                        
                        # append deviation to roughness array 
                        roughness.append(avgDev(x, xfit))
                        
                        # draw line on roughness image
                        start_point = (int(xfit[0]), int(y[0]))
                        end_point = (int(xfit[-1]), int(y[-1]))
                        cv.line(roughness_img, start_point, end_point, (255, 255, 0), 2)
                       
                    # calculate the average roughness
                    roughness = np.mean(roughness) * scale
                    
                    # append to dataset 
                    allRoughness.append(str(roughness).replace('.', ','))
                    
                    # show/save roughness image
                    if saveSummaries == True:
                        cv.imwrite(sumimgDir + '/Contour ' + ROI, roughness_img)
                    # plt.imshow(roughness_img)
                    # plt.title(ROI)
                    # plt.show()
                except: 
                    print('ISSUE WITH ' + ROI)
                   
        data.append(allRoughness)
        print(allRoughness, '\n')

# save to excel 
df = pd.DataFrame(data=list(data), columns= list(column_names), index = loi)
df.to_excel(sourcePath + 'Roughness_Thresh.xlsx')
print('All Done!')
