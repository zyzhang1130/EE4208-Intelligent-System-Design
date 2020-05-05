# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:21:22 2020

@author: Lenovo
"""


import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import time

# import the necessary packages
from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

#convolve a predefined kernel with the given image
def convolution(img, operator,flag=1):
    imgHeight, imgWidth = img.shape[0], img.shape[1]
    convolvedImg = np.zeros(img.shape)
    if flag==1:
        borderColour = cv2.BORDER_REPLICATE
    else:
        borderColour = cv2.BORDER_CONSTANT
    padding = math.floor((operator.shape[1]-1)/2)
    # padding = (operator.shape[1] - 1) // 2
    img = cv2.copyMakeBorder(img, padding, padding, padding, padding,
		borderColour)
    
    
    for x in range(padding, imgWidth+padding):
        for y in range(padding, imgHeight+padding):
            imgWindow = img[(y-padding):(y+padding+1),(x-padding):(x+padding+1)]
            result = imgWindow*operator
            convolvedPixel = int(result.sum())
            convolvedImg[(y-padding),(x-padding)] = convolvedPixel
    return convolvedImg

#a prototypical way for convolving Sobel operator with a given image. The size and value of Sobel operator is hardcoded so it is hard to change and was not used later on
def sobelOperator(img):
    # container = np.copy(img)
    m=img.shape[0]
    n=img.shape[1]
    container = np.zeros(img.shape)
    edge=np.zeros((m,n))
    size = container.shape
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            gy = -(img[i - 1][j - 1] + 2*img[i][j - 1] + img[i + 1][j - 1]) + (img[i - 1][j + 1] + 2*img[i][j + 1] + img[i + 1][j + 1])
            gx = -(img[i - 1][j - 1] + 2*img[i - 1][j] + img[i - 1][j + 1]) + (img[i + 1][j - 1] + 2*img[i + 1][j] + img[i + 1][j + 1])
            container[i][j] = np.sqrt(gx**2 + gy**2)
            normal=math.atan(gy/gx)
            # normal=math.atan2(gy,gx)
            if math.isnan(normal):
                if gy==0 and gx==0:
                    edge[i][j]=0
                else:    
                    edge[i][j]=1
            elif (gy/gx)<math.tan(math.radians(22.5)) and (gy/gx)>=math.tan(math.radians(-22.5)):
                edge[i][j] = 2
            elif (gy/gx)>math.tan(math.radians(67.5)) or (gy/gx)<=math.tan(math.radians(112.5)):
                edge[i][j] = 1
            elif (gy/gx)>=math.tan(math.radians(22.5)) and (gy/gx)<=math.tan(math.radians(67.5)):   
                edge[i][j] = 4
            else:
                edge[i][j] = 3
    return container,edge
#0: no edge
#1: 'up/down' edge
#2: 'left/right' edge
#3: 'top-left/bottom-right' edge
#4: 'top-right/bottom-left' edge

#function that finds the magnitude and direction of edges 
def EdgeandMag(img,gx,gy):
    # container = np.copy(img)
    m=img.shape[0]
    n=img.shape[1]
    container = np.zeros(img.shape)
    edge=np.zeros((m,n))
    size = container.shape
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            container[i][j] = np.sqrt(gx[i,j]**2 + gy[i,j]**2)
            normal=math.atan(gy[i,j]/gx[i,j])
            # normal=math.atan2(gy,gx)
            if math.isnan(normal):
                if gy[i,j]==0 and gx[i,j]==0:
                    edge[i][j]=0
                else:    
                    edge[i][j]=1
            elif (gy[i,j]/gx[i,j])<math.tan(math.radians(22.5)) and (gy[i,j]/gx[i,j])>=math.tan(math.radians(-22.5)):
                edge[i][j] = 2
            elif (gy[i,j]/gx[i,j])>math.tan(math.radians(67.5)) or (gy[i,j]/gx[i,j])<=math.tan(math.radians(112.5)):
                edge[i][j] = 1
            elif (gy[i,j]/gx[i,j])>=math.tan(math.radians(22.5)) and (gy[i,j]/gx[i,j])<=math.tan(math.radians(67.5)):   
                edge[i][j] = 4
            else:
                edge[i][j] = 3
    return container,edge
#0: no edge
#1: 'up/down' edge
#2: 'left/right' edge
#3: 'top-left/bottom-right' edge
#4: 'top-right/bottom-left' edge

#an alternative way of doing non-maximum-suppresion, was not used eventually
def non_maximum_suppresion(edge,img):
    size = edge.shape
    thinned = [[img[x][y] for y in range(len(img[0]))] for x in range(len(img))]
    thinned=np.asarray(thinned)
    
    for i in range(size[0]):
        thinned[i][0] = 0
        thinned[i][-1] = 0
    for i in range(size[1]):
        thinned[0][i] = 0
        thinned[-1][i] = 0        
    
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            if edge[i][j]==2: #up/down edge
                if img[i][j]<img[i+1][j] or img[i][j]<img[i-1][j]:
                    thinned[i][j]=0
            if edge[i][j]==1: #left/right edge
                if img[i][j]<img[i][j+1] or img[i][j]<img[i][j-1]:
                    thinned[i][j]=0
            if edge[i][j]==4: #top left/bottom right edge
                if img[i][j]<img[i+1][j+1] or img[i][j]<img[i-1][j-1]:
                    thinned[i][j]=0
            if edge[i][j]==3: #top right/bottom left edge
                if img[i][j]<img[i-1][j+1] or img[i][j]<img[i+1][j-1]:
                    thinned[i][j]=0
    return thinned



#use used eventually for non-maximum-suppresion
def non_maximum_suppresion2(edge,img):
    size = edge.shape
    thinned=np.zeros(size)
        
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            
            if edge[i][j]==2: #left/right edge
                if img[i][j]>=img[i+1][j] and img[i][j]>=img[i-1][j]:
                    thinned[i][j]=int(img[i][j])
                    # thinned[i+1][j]=int(img[i+1][j])
                    # thinned[i-1][j]=int(img[i-1][j])
            if edge[i][j]==1: #up/down edge
                if img[i][j]>=img[i][j+1] and img[i][j]>=img[i][j-1]:
                    thinned[i][j]=int(img[i][j])
                    # thinned[i][j+1]=int(img[i][j+1])
                    # thinned[i][j-1]=int(img[i][j-1])
            if edge[i][j]==4: #top right/bottom left edge
                if img[i][j]>=img[i+1][j+1] and img[i][j]>=img[i-1][j-1]:
                    thinned[i][j]=int(img[i][j])
                    # thinned[i+1][j+1]=int(img[i+1][j+1])
                    # thinned[i-1][j-1]=int(img[i-1][j-1])
            if edge[i][j]==3: #top left/bottom right edge
                if img[i][j]>=img[i-1][j+1] and img[i][j]>=img[i+1][j-1]:
                    thinned[i][j]=int(img[i][j])
                    # thinned[i][j]=int(img[i][j])
                    # thinned[i][j]=int(img[i][j])
    return thinned

#function to create LoG kernel give the required size and variance, not was not used         
def LoG(dim,var):
    Log=np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            Log[i,j]=-(1-((i-(dim-1)/2)**2+(j-(dim-1)/2)**2)/(2*var))*math.exp(-((i-(dim-1)/2)**2+(j-(dim-1)/2)**2)/(2*var))
    return Log

#function to create Gaussian kernel give the required size and variance    
def Gaussian(dim,var):
    Gau=np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            Gau[i,j]=math.exp(-((i-(dim-1)/2)**2+(j-(dim-1)/2)**2)/(2*var))
    Gau=Gau/sum(sum(Gau))
    return Gau

#function to create 1-d discrete Gaussian needed for constructing Sobel kernel given the required size and variance    
def G_x(dim,var):
    gx=np.zeros(dim)
    for i in range(dim):
        gx[i]=math.exp(-(i-(dim-1)/2)**2/(2*var))
    gx=gx/sum(gx)
    gx=np.reshape(gx,(1,dim))[0]
    return gx

#an alternative way of doing non-maximum-suppresion was, not used eventually
def non_maximum_suppresion3(edge,img, sobelx):
    size = edge.shape
    thinned=np.zeros(size)

    kernelSize = sobelx.shape[0]
    endpoint = int((kernelSize-1)/2)
        
    for i in range(endpoint, int(size[0] - endpoint)):
        for j in range(endpoint, int(size[1] - endpoint)):
            if edge[i,j]==2: #up/down edge
                if img[i,j]>= max(img[int(i-endpoint):int(i+endpoint+1),j]):
                    thinned[i,j]=int(img[i,j])
            if edge[i,j]==1: #left/right edge
                if img[i,j]>= max(img[i,int(j-endpoint):int(j+endpoint+1)]):
                    thinned[i,j]=int(img[i,j])
            if edge[i,j]==4: #top left/bottom right edge
                comparisonPixels = []
                for diagValues in range(int(-endpoint),endpoint+1):
                    comparisonPixels.append(img[i+diagValues,j+diagValues])
                if img[i,j]>=max(comparisonPixels):
                    thinned[i,j]=int(img[i,j])
            if edge[i,j]==3: #top right/bottom left edge
                comparisonPixels = []
                for diagValues in range(-endpoint,endpoint+1):
                    comparisonPixels.append(img[i-diagValues,j+diagValues])
                if img[i,j]>=max(comparisonPixels):
                    thinned[i,j]=int(img[i,j])
    return thinned


k=0.01
gx=G_x(3,k) #create 1-d discrete Gaussian needed for constructing Sobel kernel given the required size and variance    
sobelx=np.zeros((3,3))
sobely=np.zeros((3,3))
for i in range(1):
    #sobelx kernel
    gx=G_x(3,k)
    sobelx[0,0]=-gx[0]
    sobelx[0,1]=-gx[1]
    sobelx[0,2]=-gx[2]
    
    
    sobelx[2,0]=gx[0]
    sobelx[2,1]=gx[1]
    sobelx[2,2]=gx[2]
    
    #sobely kernel
    
    sobely[0,0]=-gx[0]
    sobely[1,0]=-gx[1]
    sobely[2,0]=-gx[2]
    
    
    sobely[0,2]=gx[0]
    sobely[1,2]=gx[1]
    sobely[2,2]=gx[2]
    k/=1.5

#the followings are different type of Sovel and Prewitt operator but was not used later on

# sobelx=np.zeros((3,3))
# sobely=np.zeros((3,3))
# sobelx[0,1]=-22.5
# sobelx[2,1]=22.5
# sobely[1,0]=-22.5
# sobely[1,2]=22.5
# for i in range(20):
#     #sobelx kernel
    
#     sobelx[0,0]=-1
#     sobelx[0,1]+=-0.5
#     sobelx[0,2]=-1
    
    
#     sobelx[2,0]=1
#     sobelx[2,1]+=0.5
#     sobelx[2,2]=1
    
#     #sobely kernel
    
#     sobely[0,0]=-1
#     sobely[1,0]+=-0.5
#     sobely[2,0]=-1
    
    
#     sobely[0,2]=1
#     sobely[1,2]+=0.5
#     sobely[2,2]=1
    
    
    
    
    
    #sobelx kernel
    # sobelx=np.zeros((5,5))
    # sobelx[0,0]=-2
    # sobelx[0,1]=-2
    # sobelx[0,2]=-3
    # sobelx[0,3]=-2
    # sobelx[0,4]=-2
    # sobelx[1,0]=-1
    # sobelx[1,1]=-1
    # sobelx[1,2]=-2
    # sobelx[1,3]=-1
    # sobelx[1,4]=-1
    # sobelx[4,0]=2
    # sobelx[4,1]=2
    # sobelx[4,2]=3
    # sobelx[4,3]=2
    # sobelx[4,4]=2
    # sobelx[3,0]=1
    # sobelx[3,1]=1
    # sobelx[3,2]=2
    # sobelx[3,3]=1
    # sobelx[3,4]=1
    # #sobely kernel
    # sobely=np.zeros((5,5))
    # sobely[0,0]=-2
    # sobely[1,0]=-2
    # sobely[2,0]=-3
    # sobely[3,0]=-2
    # sobely[4,0]=-2
    # sobely[0,1]=-1
    # sobely[1,1]=-1
    # sobely[2,1]=-2
    # sobely[3,1]=-1
    # sobely[4,1]=-1
    # sobely[0,4]=2
    # sobely[1,4]=2
    # sobely[2,4]=3
    # sobely[3,4]=2
    # sobely[4,4]=2
    # sobely[0,3]=1
    # sobely[1,3]=1
    # sobely[2,3]=2
    # sobely[3,3]=1
    # sobely[4,3]=1
    
    # #prewittx kernel
    # sobelx=np.zeros((5,5))
    # sobelx[0,0]=-7
    # sobelx[0,1]=-7
    # sobelx[0,2]=-7
    # sobelx[0,3]=-7
    # sobelx[0,4]=-7
    # sobelx[1,0]=-7
    # sobelx[1,1]=-3
    # sobelx[1,2]=-3
    # sobelx[1,3]=-3
    # sobelx[1,4]=-7
    # sobelx[4,0]=9
    # sobelx[4,1]=9
    # sobelx[4,2]=9
    # sobelx[4,3]=9
    # sobelx[4,4]=9
    # sobelx[3,0]=9
    # sobelx[3,1]=5
    # sobelx[3,2]=5
    # sobelx[3,3]=5
    # sobelx[3,4]=9
    # #prewitty kernel
    # sobely=np.zeros((5,5))
    # sobely[0,0]=-7
    # sobely[1,0]=-7
    # sobely[2,0]=-7
    # sobely[3,0]=-7
    # sobely[4,0]=-7
    # sobely[0,1]=-7
    # sobely[1,1]=-3
    # sobely[2,1]=-3
    # sobely[3,1]=-3
    # sobely[4,1]=-7
    # sobely[0,4]=9
    # sobely[1,4]=9
    # sobely[2,4]=9
    # sobely[3,4]=9
    # sobely[4,4]=9
    # sobely[0,3]=9
    # sobely[1,3]=5
    # sobely[2,3]=5
    # sobely[3,3]=5
    # sobely[4,3]=9
    
    # #sobelx kernel
    # sobelx=np.zeros((7,7))
    # sobelx[0,0]=-2
    # sobelx[0,1]=-2
    # sobelx[0,2]=-4
    # sobelx[0,3]=-2
    # sobelx[0,4]=-2
    # sobelx[0,3]=-2
    # sobelx[0,4]=-2
    # sobelx[2,0]=2
    # sobelx[2,1]=2
    # sobelx[2,2]=4
    # sobelx[2,3]=2
    # sobelx[2,4]=2
    # #sobely kernel
    # sobely=np.zeros((7,7))
    # sobely[0,0]=-2
    # sobely[1,0]=-2
    # sobely[2,0]=-4
    # sobely[3,0]=-2
    # sobely[4,0]=-2
    # sobely[0,2]=2
    # sobely[1,2]=2
    # sobely[2,2]=4
    # sobely[3,2]=2
    # sobely[4,2]=2
    
    
   
    img = cv2.imread("cana.jpg")  #read image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #read as grey scale
    
    blur = convolution(img, Gaussian(3,2)) #convole image with Gaussian kernel
    # img2,edge = sobelOperator(blur) #the prototypical way of convolving input iamge with hard coded Sobel operator. Not used later on 
    gx=convolution(blur, sobelx) #convole image with Sobel-x kernel
    gy=convolution(blur, sobely)#convole image with Sobelyy kernel
    
    img2,edge=EdgeandMag(img,gx,gy) #find the magnitude and direction of edges 
    
    thinned=non_maximum_suppresion2(edge,img2) #apply non-maximum suppresion
    
    #apply 2 thresholding to the image as preparation for hysterisis thresholding
    size = thinned.shape
    thresh1=np.copy(thinned)
    thresh2=np.copy(thinned)
    for i in range(size[0]):
            for j in range(size[1]):
                if thresh1[i][j]>40:
                    thresh1[i][j]=255
                else:
                    thresh1[i][j]=0
                    
                    
                    
    for i in range(size[0]):
            for j in range(size[1]):
                if thresh2[i][j]>10:
                    thresh2[i][j]=255
                else:
                    thresh2[i][j]=0
    
                    
                    
                    
    #display images at different stages of Canny edge detection             
    plt.figure()
    plt.show()
    plt.imshow(img,cmap='gray', vmin=0, vmax=255)                
                    
    plt.figure()
    plt.show()
    plt.imshow(blur,cmap='gray', vmin=0, vmax=255)
    
    plt.figure()
    plt.show()
    plt.imshow(img2,cmap='gray', vmin=0, vmax=255)
    
    plt.figure()
    plt.show()
    plt.imshow(thinned,cmap='gray', vmin=0, vmax=255)
    
    plt.figure()
    plt.show()
    plt.imshow(thresh1,cmap='gray', vmin=0, vmax=255)
    
    plt.figure()
    plt.show()
    plt.imshow(thresh2,cmap='gray', vmin=0, vmax=255)
    
    #hysterisis thresholding
    flag=0
    counter=0
    result = np.copy(thresh1)
    while flag==0:
        ref = np.copy(result)
        for i in range(size[0]):
                for j in range(size[1]):
                    if result[i][j]!=0:
                        if thresh2[i-1][j-1]!=0:
                            result[i-1][j-1]=thresh2[i-1][j-1]
                        if thresh2[i-1][j]!=0:
                            result[i-1][j]=thresh2[i-1][j]
                        if thresh2[i-1][j+1]!=0:
                            result[i-1][j+1]=thresh2[i-1][j+1]
                        if thresh2[i][j-1]!=0:
                            result[i][j-1]=thresh2[i][j-1]
                        if thresh2[i][j+1]!=0:
                            result[i][j+1]=thresh2[i][j+1]
                        if thresh2[i+1][j-1]!=0:
                            result[i+1][j-1]=thresh2[i+1][j-1]
                        if thresh2[i+1][j]!=0:
                            result[i+1][j]=thresh2[i+1][j]
                        if thresh2[i+1][j+1]!=0:
                            result[i+1][j+1]=thresh2[i+1][j+1]
        counter+=1
        a=(ref==result)
        b=a.all()
        if b:
            flag=1
            
    #display final output    
    plt.figure()
    plt.show()
    plt.imshow(result,cmap='gray', vmin=0, vmax=255)
