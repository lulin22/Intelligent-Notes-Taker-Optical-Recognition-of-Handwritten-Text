# coding: utf-8

# In[ ]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


import math

import os
import heapq

from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize


# In[ ]:



image = cv2.imread('t8.jpg')

#grayscale
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# convolusion using gaussian filter 
#gray=cv2.GaussianBlur(gray, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) 
# #binary
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
cv2.imshow('second',thresh)
cv2.waitKey(0)

#dilation
kernel = np.ones((5,100), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
im2,ctrs = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#sort contours
sorted_ctrs = sorted(im2, key=lambda ctr: cv2.boundingRect(ctr)[0])


# In[ ]:


roi=[]
for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi.append(image[y:y+h, x:x+w])

    # show ROI9
   # cv2.imshow('segment no:'+str(i),roi[i])
    cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
    #cv2.waitKey(0)


# In[ ]:


all_sumCols=[]
final_Img=[]
for subimg in roi:
    subimg=cv2.cvtColor(subimg,cv2.COLOR_BGR2GRAY)
    height, weight = subimg.shape

   # "Return a list containing the sum of the pixels in each column"
    sumCols = []
    for j in range(weight):
        col = subimg[0:height, j:j+1] # y1:y2, x1:x2
        sumCols.append(np.sum(col))
    
    #find the cutting points for subimg
    maxSum=max(sumCols) 
#    temp=0
    segImg=[]
    subl=0
    subr=0
    flag=[]
    for s in range(weight):
        col = subimg[0:height, s:s+1]
        if np.sum(col)>0.95*maxSum:    
#             segImg.append(subimg[0:height, temp:j]) # should only append black 
#             #print segImg
#             temp=j
            flag.append(0)
            if s>0:
                if flag[s-1]==1:
                    subr=s
                    segImg.append(subimg[0:height, subl:subr])
        else:
            flag.append(1)
            if s>0:
                if flag[s-1]==0:
                    subl=s
                    if subl-subr>0.2*height:
                        segImg.append(subimg[0:height, subr:subl])
    all_sumCols.append(sumCols)
    final_Img.append(segImg)


# In[ ]:


# for i in range(len(final_Img)):
#     for j in range(len(final_Img[i])):
#         cv2.imshow('seg img'+str(i)+str(j),final_Img[i][j])
#         cv2.waitKey(0)


# In[ ]:


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),0)
        if img is not None:
            images.append(img)
    return images


# In[ ]:


uniset=load_images_from_folder('alphabet2')

query=final_Img


# In[ ]:


def lbp_histo(img):
    r=3
    n=8*r
    histo=np.zeros((len(img),n+4))
    for m in range(len(img)):
        #img_gray=cv2.cvtColor(img[m],cv2.COLOR_BGR2GRAY)
        img_gray=img[m]
        lbp=local_binary_pattern(img_gray,n,r,method='uniform')
        x=itemfreq(lbp.ravel())
        for i in range(len(x)):
            histo[m][i]=x[i,1]/sum(x[:,1])
    return histo  


# In[ ]:


image = cv2.imread('t2.png')

#grayscale
#gray=cv2.cvtColor(unise,cv2.COLOR_BGR2GRAY)
lbp=local_binary_pattern(uniset[3],24,3,method='uniform')
x=itemfreq(lbp.ravel())

len(x)


# In[ ]:


# define distance function 
def dist(v1,v2):
    s=0;
    for i in range(len(v1)):
        s+=(v1[i]-v2[i])*(v1[i]-v2[i])
    return math.sqrt(s)


# In[ ]:


# calculate lbp histogram for query and uniset 
q=[]
query_lbp=[]

for i in range(len(query)):
    q.append(query[i])
    query_lbp.append(lbp_histo(q[i]))
    
uniset_lbp=lbp_histo(uniset)


# In[ ]:


differ_lbp=np.zeros((len(query),500,len(uniset)))# the maximum number of characters in a single line
#differ_lbp=[[][][]]
for k in range(len(query)):
    for m in range(len(q[k])):
        for i in range(len(uniset)):
            differ_lbp[k][m][i]=dist(query_lbp[k][m],uniset_lbp[i])
        


# In[ ]:


unibase=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','$']


# In[ ]:


# text=[]
# for k in range(len(query)):
#     for m in range(len(q[k])):
#         a=differ_lbp[k][m,:]
#         text.extend()
# print text


# In[ ]:


# cv2.imshow('3',image)
# cv2.waitKey(0)


# In[ ]:


print len(uniset_lbp)


# In[ ]:


# for k in range(len(query)):
#     for m in range(len(q[k])):
#         for i in range(len(uniset)):
#             differ_lbp[k][m][i]=dist(query_lbp[k][m],uniset_lbp[i])
        
        
text=[]
diff=[]
for i in range(len(query)):
    diff.append([])
    text.append([])
    for j in range(len(query[i])):
        diff[i].append([])
        text[i].append([])
        for k in range(len(uniset)):
            dist2=dist(query_lbp[i][j],uniset_lbp[k])
            diff[i][j].append([dist2,k])
            #sml2=simi(query_lbp[i],uniset_lbp[j])
            #siml2[i].append([sml2,i,j])

        diff[i][j].sort(key=lambda x:x[0], reverse=False)
#        siml2[i][j].sort(key=lambda x:x[0], reverse=True)
        char=unibase[diff[i][j][0][1]]
        text[i][j].append(char)
    print text[i]


# In[ ]:


# for i in range(len(final_Img)):
#     for j in range(len(final_Img[i])):
#         cv2.imshow('seg img'+str(i)+str(j),final_Img[i][j])
#         cv2.waitKey(0)
