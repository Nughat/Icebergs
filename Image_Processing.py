import rasterio
import sys
from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.mask import mask
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import date
import os
import glob
from pyproj import Transformer
from scipy import stats
import matplotlib.ticker as plticker

def deleteNoDatas(directory):
    adir = directory
    for file in os.scandir(adir):
        #print(file.path)
        ice = cv2.imread(file.path)
        #plt.imshow(ice)
        #plt.show()
        if np.all(ice==0) == True:
            print(file.path)
            plt.imshow(ice)
            plt.show()
            #os.remove(file.path)
    print('done')
#deleteNoDatas(r'/net/aeh/nuzhat_data/a68a')
def area(mask,img):
    height, width, channels = img.shape
    area = cv2.countNonZero(mask)*(300*300)/(width*height)
    #print('AREA',area)
    return area
    
def mods367(directory):
    adir = directory
    trackAreas = []
    for file in adir:
        #print(file)
        rimg = rasterio.open(file)
        band1 = rimg.read(1)
        img = cv2.imread(file)
        #print('original')
        #show(rimg)

        ret,thresh4 = cv2.threshold(band1,210,255,cv2.THRESH_BINARY)
        #print('first threshold')
        #plt.imshow(thresh4,'gray')
        #plt.show()


        # Perform an area filter on the binary blobs:
        componentsNumber, labeledImage, componentStats, componentCentroids = \
        cv2.connectedComponentsWithStats(thresh4, connectivity=8)
        # Set the minimum pixels for the area filter:
        minArea = 800
        # Get the indices/labels of the remaining components based on the area stat
        # (skip the background component at index 0)
        remainingComponentLabels = [i for i in range(1, componentsNumber) if componentStats[i][4] >= minArea]
        # Filter the labeled pixels based on the remaining labels,
        # assign pixel intensity to 255 (uint8) for the remaining pixels
        filteredImage = np.where(np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8')
        #print('removed white blobs')
        #plt.imshow(filteredImage,'gray')
        #plt.show()

        kernel = np.ones((5,5), np.uint8)
        img_dilation = cv2.dilate(filteredImage, kernel, iterations=2)
        #img_dilation = cv2.dilate(thresh4, kernel, iterations=2)
        blur = cv2.blur(img_dilation,(15,15))

        rets, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #print('thresh')
        #plt.imshow(thresh,'gray')
        #plt.show()
        contours1, hierarchy1 = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)


        i = cv2.drawContours(img, contours, -1, (0,255,0), 3)
        #plt.imshow(i)
        #plt.show()

        p = cv2.drawContours(img, contours1, -1, (0,255,0), 3)
        #plt.imshow(p)
        #plt.show()

        cnt = max(contours, key=cv2.contourArea)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [cnt],-1, 255, -1)
        res = cv2.bitwise_and(img, img, mask=mask)
        #plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
        #plt.show()
        
        trackAreas.append(area(mask,img)) 
    return trackAreas