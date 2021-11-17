# coding: utf-8

# In[15]:




# In[1]:


import cv2
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


import image
image = cv2.imread('/Users/linlu/Desktop/1.jpg')
#cv2.imshow('orig',image)
#cv2.waitKey(0)

#grayscale
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#cv2.imshow('gray',gray)
#cv2.waitKey(0)


# In[4]:


# #binary
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
cv2.imshow('second',thresh)
cv2.waitKey(0)
#plt.plot(thresh)
#plt.show()


# In[ ]:


# In[5]:


#dilation
kernel = np.ones((5,100), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)

#cv2.imshow('dilated',img_dilation)
#cv2.waitKey(0)


# In[6]:


im2,ctrs = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# In[7]:



#cv2.drawContours(image,im2,-1,(0,255,0),3)


# In[8]:


#cv2.imshow("Contour",image)


# In[9]:


#sort contours
sorted_ctrs = sorted(im2, key=lambda ctr: cv2.boundingRect(ctr)[0])


# In[10]:


#for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
  #  x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
 #   roi = image[y:y+h, x:x+w]

    # show ROI
   # cv2.imshow('segment no:'+str(i),roi)
    #cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
    #cv2.waitKey(0)

#cv2.imshow('marked areas',image)
#cv2.waitKey(0)


# In[11]:


#for i, ctr in enumerate(sorted_ctrs):
 #   x, y, w, h = cv2.boundingRect(ctr)
  #  roi = image[y:y+h, x:x+w]
   # hist = cv2.calcHist([roi],[0],None,[256],[0,256])
    #cv2.imshow('histogram no:'+str(i),hist)
    #plt.plot(histr,color = col)
    #plt.xlim([0,256])
#plt.show()
    
#cv2.waitKey(0)


# In[ ]:


# for i, ctr in enumerate(sorted_ctrs):
#     x, y, w, h = cv2.boundingRect(ctr)
#     roi = image[y:y+h, x:x+w]
#     roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
#     cv2.imshow('segment no:'+str(i),roi)

#    # "Return a list containing the sum of the pixels in each column"
#     sumCols = []
#     for j in range(w):
#         col = roi[0:h, j:j+1] # y1:y2, x1:x2
#         sumCols.append(np.sum(col))
        
#     plt.plot(sumCols)
#     plt.ylabel('line number:'+str(i))
#     plt.show()
#     cv2.waitKey(0)
# sumCols


# In[ ]:


#import matplotlib.pyplot as plt
#plt.plot(sumCols)
#plt.ylabel('some numbers')
#plt.show()


# In[ ]:


#maxSum=max(sumCols)


# In[12]:


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

#cv2.imshow('marked areas',image)
#cv2.waitKey(0)


# In[ ]:


x,y,z = image.shape


# In[ ]:


print image.shape


# In[13]:


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
        if np.sum(col)>maxSum-10:    
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
        
            
    #print sumCols  
   # plt.plot(sumCols)
    #plt.show()
    #cv2.waitKey(0)
    
    #cv2.imshow('yun',segImg[0])
    #cv2.waitKey(0)
    #print segImg
    all_sumCols.append(sumCols)
    final_Img.append(segImg)
    
    #plt.plot(sumCols)
    #plt.ylabel('line number:'+str(i))
    #plt.show()
    #cv2.waitKey(0)


# In[ ]:


for subimg in roi:
    subimg=cv2.cvtColor(subimg,cv2.COLOR_BGR2GRAY)
    height, weight = subimg.shape
    col = subimg[0:height,0:200]
    print col
    cv2.imshow('l',col)
    cv2.waitKey(0)


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
    print maxSum, height


# In[ ]:


len(final_Img)


# In[ ]:


len(final_Img[0])


# In[ ]:


for i in range(len(final_Img)):
    for j in range(len(final_Img[i])):
        cv2.imshow('seg img'+str(i)+str(j),final_Img[i][j])
        cv2.waitKey(0)
    


# In[17]:


range(final_Img)
