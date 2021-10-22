import rasterio
from rasterio.plot import show
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import datetime
from datetime import datetime
from pyproj import Transformer
from scipy import stats
import pandas as pd
import matplotlib.ticker as plticker

from functools import partial    
import shapely
import shapely.ops as ops
from shapely.geometry.polygon import Polygon
from shapely.ops import orient
from shapely import wkt #geod
from pyproj import Geod, CRS
from pyproj import Proj

import pickle

def area2(cont,matrix,retCon =  False):
    #T0 = matrix
    with rasterio.open(matrix, 'r') as r:
        T0 = r.transform
    temp = []
    for i in cont:
        temp.append(T0*(i[0][1],i[0][0]))
    #print(temp[0])
    ps_to_latlon = []
    transformer = Transformer.from_crs("epsg:3031","epsg:4326") 
    for i in temp:
        (x,y) = transformer.transform(i[0],i[1])
        ps_to_latlon.append((x,y))
    #print(ps_to_latlon[0])
    geom = Polygon(ps_to_latlon)
    geod = Geod(ellps="WGS84")
    poly_area, poly_perimeter = geod.geometry_area_perimeter(orient(geom))
    if retCon == True:
        return poly_area/1e+6,ps_to_latlon
    else:
        return poly_area/1e+6
    
def area3(cont,matrix,retCon =  False):
    #T0 = matrix
    with rasterio.open(matrix, 'r') as r:
        T0 = r.transform
    temp = []
    for i in cont:
        temp.append(T0*(i[0][1],i[0][0]))
    geom = Polygon(temp)
    geod = Geod(ellps="WGS84")
    poly_area, poly_perimeter = geod.geometry_area_perimeter(orient(geom))
    if retCon == True:
        return poly_area/1e+6,temp
    else:
        return poly_area/1e+6

def iceStats(dataSet):
    df = pd.DataFrame(list(dataSet))
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    med = df.median()
    IQR = Q3 - Q1
    lower_bound = Q1 -(1.5 * IQR) 
    upper_bound = Q3 +(1.5 * IQR) 
    print('MED',med)
    print('LOWER',lower_bound)
    print('UPPER',upper_bound)
    print('IQR',IQR)
    print('\n')
    
def findtooLarge(dataSet):   
    tooLarge = [i for i,j in dataSet.items() if j > 10000]
    return tooLarge

def timeSeries(d,aList,lbl):
    plt.style.use("seaborn")
    plt.figure(figsize=(8, 6), dpi=80)
    plt.scatter(d,aList,label=lbl)
    plt.legend()
    plt.xticks(rotation=45)
    plt.ylim(3000, 10000)
    
    plt.boxplot(np.array(aList))
    
    np.array
    plt.show()

def getdates(dictionary):
    dates = []
    chenk = []
    for i,j in dictionary.items():
        r = i[::-1]
        #print(r)
        d = r[0:10]
        #print(d)
        d = d[::-1]
        #print(d)
        date_time_obj = datetime.strptime(d, '%Y-%m-%d').date()
        #print(date_time_obj)
#         print(i,j)
#         print(date_time_obj,j)
        chenk.append((date_time_obj,j))
        chenk.sort()
        #dates.append(date_time_obj)
    return chenk

def isolate(directory,layerName=None,display=False,setThresh=None,areaThresh=None,cutOff=None,reprocess=False,geo=False,retCnt = False):
    """Isolate iceberg from its surroundings.
    """
    if reprocess == True and type(directory) != str:
        adir = directory
    else:
        adir = glob.glob(directory)
    trackAreas = {} #this is now a dictionary
    for file in adir:
        rimg = rasterio.open(file)
        img = cv2.imread(file)
        if display == True:
            print(file)
            print('ORIGINAL')
            show(rimg)        
        if layerName == "red":
            band = rimg.read(1)         
        if layerName == "blue":
            band = rimg.read(3)    
        elif layerName == None:
            band = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
            
        if setThresh != None: #If image is being reprocessed, the threshold can be adjusted for better results
            ret,thresh4 = cv2.threshold(band,setThresh,255,cv2.THRESH_BINARY) #Eg: 200 instead of 210 makes quite a difference for some images
        else:   
            ret,thresh4 = cv2.threshold(band,210,255,cv2.THRESH_BINARY)
        
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

        kernel = np.ones((5,5), np.uint8)
        img_dilation = cv2.dilate(filteredImage, kernel, iterations=2)
        #img_dilation = cv2.dilate(thresh4, kernel, iterations=2)
        blur = cv2.blur(img_dilation,(15,15))

        rets, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #contours1, hierarchy1 = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) #obrain contours
        
        if (contours):
            if display == True:
                allContours = cv2.drawContours(img, contours, -1, (0,255,0), 3) #draw all contours on image
                print('CONTOURED')
                plt.imshow(cv2.cvtColor(allContours, cv2.COLOR_BGR2RGB))
                plt.show()
            
            #reprocessing : calculate contours of area
            if reprocess == True:
#                 if areaThresh=None or cutOff=None: #throw error
#                     print("")
                cnt = contours[0]
                for i in range(len(contours)):
                    
                    icnt = contours[i]
                    if i == 0:
                        if geo == True:
                            BergArea,coords = area3(cnt,file,retCon=True)
                        else:
                            BergArea,coords = area2(icnt,file,retCon = True)
                        #BergArea,coords = area2(icnt,file,retCon = True)
                        print(BergArea)
                        howClose = abs(BergArea-areaThresh) 
                        if display == True:
                            h, w = img.shape[:2]
                            mask = np.zeros((h, w), np.uint8)
                            cv2.drawContours(mask, [icnt],-1, 255, -1)
                            res = cv2.bitwise_and(img, img, mask=mask)
                            print('CONTOUR',i)
                            plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
                            plt.show()
                            print('AREA',BergArea)
                    else:
                        if geo == False:
                            if abs(area2(icnt,file)-areaThresh) < howClose and area2(icnt,file) < cutOff:
                                BergArea,coords  = area2(icnt,file,retCon = True)
                                howClose = abs(BergArea-areaThresh)
                                cnt = icnt
                                if display == True:
                                    h, w = img.shape[:2]
                                    mask = np.zeros((h, w), np.uint8)
                                    cv2.drawContours(mask, [icnt],-1, 255, -1)
                                    res = cv2.bitwise_and(img, img, mask=mask)
                                    print('CONTOUR',i)
                                    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
                                    plt.show()
                                    print('AREA',BergArea)
                        else:
                            if abs(area3(icnt,file)-areaThresh) < howClose and area3(icnt,file) < cutOff:
                                BergArea,coords  = area3(icnt,file,retCon = True)
                                howClose = abs(BergArea-areaThresh)
                                cnt = icnt
                                if display == True:
                                    h, w = img.shape[:2]
                                    mask = np.zeros((h, w), np.uint8)
                                    cv2.drawContours(mask, [icnt],-1, 255, -1)
                                    res = cv2.bitwise_and(img, img, mask=mask)
                                    print('CONTOUR',i)
                                    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
                                    plt.show()
                                    print('AREA',BergArea)
            else:
                cnt = max(contours, key=cv2.contourArea) #find max contour
            
                if geo == True:
                    BergArea,coords  = area3(cnt,file,retCon = True)
                else:
                    BergArea,coords  = area2(cnt,file,retCon = True)
                if display == True:
                    h, w = img.shape[:2]
                    mask = np.zeros((h, w), np.uint8)
                    cv2.drawContours(mask, [cnt],-1, 255, -1)
                    #BergArea = area(mask,img)
                    trackAreas[file] = BergArea
                    print('MASKED')
                    res = cv2.bitwise_and(img, img, mask=mask)
                    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
                    plt.show()
                    print('AREA',BergArea)
            
            trackAreas[file] = BergArea   
            
#         else:
#             print(file)
#             print('The contours of this image cannot be identified. Please check it.')
      
    if retCnt == True:
        return trackAreas,cnt,coords
    else:
        return trackAreas