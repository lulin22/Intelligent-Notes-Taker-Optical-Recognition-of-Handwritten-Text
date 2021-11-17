# coding: utf-8

# In[18]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from __future__ import division
import os



from PIL import Image
image = cv2.imread('/Users/linlu/Desktop/1.jpg')
#cv2.imshow('orig',image)
#cv2.waitKey(0)

#grayscale
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#cv2.imshow('gray',gray)
#cv2.waitKey(0)



# #binary
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)


# In[19]:


# In[5]:


#dilation
kernel = np.ones((5,100), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)

im2,ctrs = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#sort contours
sorted_ctrs = sorted(im2, key=lambda ctr: cv2.boundingRect(ctr)[0])


roi=[]
for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi.append(image[y:y+h, x:x+w])

    # show ROI9
    cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)

x,y,z = image.shape

#print image.shape

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
    segImg=[]
    subl=0
    subr=0
    flag=[]
    for s in range(weight):
        col = subimg[0:height, s:s+1]
        if np.sum(col)>0.98*maxSum:    
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
                        segImg.append(sub[0:height,subr:subl])
        
    all_sumCols.append(sumCols)
    final_Img.append(segImg)


# In[ ]:



for i in range(len(final_Img)):
    for j in range(len(final_Img[i])):
        cv2.imshow('seg img'+str(i)+str(j),final_Img[i][j])
        cv2.waitKey(0)
    



# In[ ]:


uniset=[]
for i in range(200):#size of database
    uniset_name ='{:0>4d}'.format(i) + '.jpg'
    uniset.append(cv2.imread(uniset_name))



# In[ ]:


query=final_Img


def dist (v1,v2):
    val=0;
    for i in range(len(v1)):
        val+=math.pow(v1[i]-v2[i],2)
    return math.sqrt(val)

def simi (v1,v2):
    val1=0;
    val2=0;
    val3=0
    for i in range(len(v1)):
        val1 +=math.pow(v1[i],2)
        val2 +=math.pow(v2[i],2)
        val3 +=(v1[i]*v2[i])
    val =val3/(math.sqrt(val1)*math.sqrt(val2))
    return val

imlbp=[]
for m in range(len(uniset)):
    print "Process uniset",m,"..."
    row = uniset[m].shape[0]
    col = uniset[m].shape[1]
    pixels=row*col
    grey=np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            idx=0
            for k in range(3):
                idx+= uniset[m][i,j,k]
            grey[i][j]=idx
    gimage=grey
    lbp=np.zeros((row-2,col-2))
    for i in range(1,row-1):
        for j in range(1,col-1):
            idx=np.zeros(8)
            if grey[i-1][j-1]>=grey[i][j]:
                idx[0]=1
            if grey[i-1][j]>=grey[i][j]:
                idx[1]=1
            if grey[i-1][j+1]>=grey[i][j]:
                idx[2]=1
            if grey[i][j+1]>=grey[i][j]:
                idx[3]=1
            if grey[i+1][j+1]>=grey[i][j]:
                idx[4]=1
            if grey[i+1][j]>=grey[i][j]:
                idx[5]=1
            if grey[i+1][j-1]>=grey[i][j]:
                idx[6]=1
            if grey[i][j-1]>=grey[i][j]:
                idx[7]=1
            loc=0
            for k in range(8):
                loc+=idx[k]*math.pow(k,2)
            lbp[i-1][j-1]=loc
    
    imlbps=[x for x in np.histogram(lbp,bins=48)[0]]
    imlbp.append(imlbps)
print ('lbp done for candidate')
#    return grey


qlbp=[]
for m in range(len(query)):
    print "Process query",m,"..."
    row = query[m].shape[0]
    col = query[m].shape[1]
    pixels=row*col
    grey=np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            idx=0
            for k in range(3):
                idx+= query[m][i,j,k]
            grey[i][j]=idx
    gquery=grey
    lbp=np.zeros((row-2,col-2))
    for i in range(1,row-1):
        for j in range(1,col-1):
            idx=np.zeros(8)
            if grey[i-1][j-1]>=grey[i][j]:
                idx[0]=1
            if grey[i-1][j]>=grey[i][j]:
                idx[1]=1
            if grey[i-1][j+1]>=grey[i][j]:
                idx[2]=1
            if grey[i][j+1]>=grey[i][j]:
                idx[3]=1
            if grey[i+1][j+1]>=grey[i][j]:
                idx[4]=1
            if grey[i+1][j]>=grey[i][j]:
                idx[5]=1
            if grey[i+1][j-1]>=grey[i][j]:
                idx[6]=1
            if grey[i][j-1]>=grey[i][j]:
                idx[7]=1
            loc=0
            for k in range(8):
                loc+=idx[k]*math.pow(k,2)
            lbp[i-1][j-1]=loc
    
    qlbps=[x for x in np.histogram(lbp,bins=48)[0]]
    qlbp.append(qlbps)
print ('lbp done for query')



color=[[]]
lbp=[[]]
diff=[]
diff1=[]
diff2=[]
siml=[]
siml1=[]
siml2=[]

#print len(qlbp)
text=[]
for i in range(len(query)):
    diff.append([])
    diff1.append([])
    diff2.append([])
    siml.append([])
    siml1.append([])
    siml2.append([])
    print "for query", i,"..."
    for j in range(len(uniset)):

        dist2=dist(qlbp[i],imlbp[j])
        diff2[i].append([dist2,i,j])
        sml2=simi(qlbp[i],imlbp[j])
        siml2[i].append([sml2,i,j])

    diff2[i].sort(key=lambda x:x[0], reverse=False)
    siml2[i].sort(key=lambda x:x[0], reverse=True)
    text.append(diff2[i][0])
        
print text
