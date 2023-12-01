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
def findContour(img, morph):
    '''
    28/30 success rate if implemented correctly
    img: image array
    morph: True or False, indicates whether morphology closing is used 
    contours: image array of contours 
    '''
    if morph == True:
        # morphology closed means dilating foreground pixels and then eroding them
        # kernel determines the thickness of the dilation/erosion 
        kernel = cv.getStructuringElement(cv.MORPH_RECT,(6,6))
        closed = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
        blurred = ndimage.gaussian_filter(closed, 2, mode='nearest')
        
    elif morph == False: 
        blurred = ndimage.gaussian_filter(img, 2, mode='nearest')
    
    if np.ndim(blurred) != 2:
        blurred = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY) 
    
    # otsu thresholding picks the threshold values based on a normalized histogram
    ret, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU) 
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
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

def findcntOI(img, morph):
    cnts = findContour(img, morph)
    cntsOI = np.argsort(findAreas(cnts))[-5:-2] 
    return cntsOI

def checkContour(cnts, cntsOI, expected):
    '''
    every actual height must be at least 90% the expected height 

    '''
    cnt_heights = findHeights(cnts, cntsOI)
    for height in cnt_heights: 
        if 0.9 * expected <= height:
            continue
        else: 
            return False
    return True

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
scale = 5.88 git 
sourcePath = '//wp-oft-nas/HiWis/GM_Dawn_Zheng/Arvid/Magnesium Walls for Dawn/'
inDir = sourcePath + 'Post Processed'
cntNames = []
allRoughness = []
acceptedFileTypes = ["png"] # add more as needed
saveSummaries = True

# for every picture in your directory, to extract the contours for analysis 
for sample in os.listdir(inDir): 
    if( '.' in sample and sample.split('.')[-1] in acceptedFileTypes):
        f = inDir + '/' + sample
        print('Processing ' + sample)
        sampleNumber = sample[:-4]
        
        # make output directory if it doesn't exist already 
        outDir = sourcePath + ' Output/' + sampleNumber + '_output'
        if not os.path.exists(outDir):
            os.makedirs(outDir)
        
        # read img
        img = cv.imread(f) 
        
        # extract img height for morph checker
        img_height = np.shape(img)[0] 
        
        # copy img for summary img
        summary_image = img.copy() 
    
        
        # try morphology first because higher success rate of 24/30
        # no morphology has a 19/30 success rate 
        cnts = findContour(img, True)
        
        # sorts areas from smallest to largest
        # extracts 3rd, 4th, and 5th largest contours
        cntsOI = np.argsort(findAreas(cnts))[-5:-2] 
       
        
        # checking contours, excluding morphology if contours are too short
        if checkContour(cnts, cntsOI, img_height) == False:
            cnts = findContour(img, False)
            cntsOI = np.argsort(findAreas(cnts))[-5:-2] 
        
        
        # extracting contours
        ROI_number = 0
        for i in cntsOI:
            x,y,w,h = cv.boundingRect(cnts[i])
            ROI = img[y:y+h, x:x+w]
            cv.imwrite('{}/{} ROI{}'.format(outDir, sampleNumber, ROI_number) + '.png', ROI)
            cv.rectangle(summary_image,(x,y),(x+w,y+h),(255,255,0),2) # also add label later on
            ROI_number += 1
       
        
        # make summary directory if it doesn't exist 
        sumimgDir = sourcePath + '/summary_imgs'
        if not os.path.exists(sumimgDir):
            os.makedirs(sumimgDir)
        
        
        #show and save extraction summary image
        if saveSummaries == True:
            cv.imwrite(sumimgDir + '/' + sample, summary_image)
        plt.imshow(summary_image)
        plt.title(sample + ' ROI')
        plt.show()
        
        
        # begin roughness script   
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
                    
                    # append contour name to indices if everything else works 
                    cntNames.append(ROI)
                    
                    # show/save roughness image
                    if saveSummaries == True:
                        cv.imwrite(sumimgDir + '/Contour ' + ROI, roughness_img)
                    plt.imshow(roughness_img)
                    plt.title(ROI)
                    plt.show()
                except: 
                    print('ISSUE WITH ' + ROI)

# save to excel 
df = pd.DataFrame(data=list(allRoughness), columns=['Roughness'], index=cntNames)
df.to_excel(sourcePath + 'Roughness.xlsx')
print('All Done!')