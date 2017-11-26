# -*- coding: utf-8 -*-

import cv2
import numpy as np
from skimage.feature import ORB, match_descriptors
from scipy.spatial.distance import euclidean
import random
from PIL import Image


'''extract features'''
img1 = Image.open('hopkins1.jpg')
img2 = Image.open('hopkins2.jpg')
img1gray = np.array(img1.convert('L'))
img2gray = np.array(img2.convert('L'))
row, col = img1gray.shape
image2 = cv2.imread('hopkins2.jpg')

'''extract features and create descriptors'''
detector_extractor1 = ORB(n_keypoints=500)
detector_extractor1.detect_and_extract(img1gray)
des1 = detector_extractor1.descriptors
co1 = detector_extractor1.keypoints
detector_extractor2 = ORB(n_keypoints=500)
detector_extractor2.detect_and_extract(img2gray)
des2 = detector_extractor2.descriptors
co2 = detector_extractor2.keypoints

'''match the descriptors'''
matches = []
for i in range(len(des1)):
    Gmax = 1000;
    Lmax = 1000;
    
    for j in range(len(des2)):
        diff = euclidean(des1[i],des2[j])
        if diff < Gmax:
            Lmax = Gmax
            Gmax = diff
            k = j
        if  Gmax < diff < Lmax:
            Lmax = diff
            
    if Gmax/Lmax <=0.835:
        matches.append([i,k])
        
'''Visualize the matches'''
combine = np.concatenate((img1,img2), axis = 1)
for i in range(len(matches)):
    ind1 = matches[i][0]
    ind2 = matches[i][1]
    y1,x1 = co1[ind1]
    y2,x2 = co2[ind2]
    x2 = x2 + col
    cv2.circle(combine,(int(x1),int(y1)),3,(0,0,255),-1)
    cv2.circle(combine,(int(x2),int(y2)),3,(0,0,255),-1)
    cv2.line(combine,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
cv2.imwrite('match.png', combine)
cv2.imshow('match_image',combine)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''Define the fundamential matrix computation'''
def compute(rand,matches,des1,des2):
    A = np.zeros((len(rand),9))
    for i in range(len(rand)):
        ind1 = rand[i]
        o1,p1 = matches[ind1]
        vl,ul = des1[o1]
        vr,ur = des2[p1]
        A[i] = [ul*ur, ul*vr, ul, vl*ur, vl*vr, vl, ur, vr, 1]
    
    AA = np.dot(np.transpose(A),A)
    w,v = np.linalg.eig(AA)
    sortlist = v[:,w.argsort()]
    H = sortlist[:,0]
    Hgood = np.array([[H[0], H[1], H[2]],
                      [H[3], H[4], H[5]],
                      [H[6], H[7], H[8]]])
    return Hgood

'''RANSAC'''
goodCount = 0
goodmatches = []
l = len(matches)

for i in range(100):
    count = 0
    randn = random.sample(range(l - 1), 8)
    F = compute(randn, matches, co1, co2)
    indlist = []
    for j in range(l):
        o,p = matches[j]
        vl,ul = co1[o]
        vr,ur = co2[p]
        c1h = np.array([ul,vl,1])
        c2h = np.array([ur,vr,1])
        r = np.dot(c1h, np.dot(F, np.transpose(c2h)))
        diff = abs(r)
        if diff < 0.05:
            count += 1
            #print('im here')
            indlist.append(j)
            #print(indlist)
    if count > goodCount:
        goodCount = count
        goodmatches = indlist

FinalF = compute(goodmatches, matches, co1, co2)

'''Plot the epipolar lines on the right image, with corresponding points'''
image1 = cv2.imread('hopkins1.jpg')
image2 = cv2.imread('hopkins2.jpg')

randomPoint = random.sample(goodmatches, 8)
for i in range(8):
    ind = randomPoint[i]
    o,p = matches[ind]
    y1,x1 = co1[o]
    y2,x2 = co2[p]
    c1h = np.array([x1, y1, 1])
    a,b,c = np.dot(c1h,FinalF)
    x0y = int(- c / b)
    y0x = int(- c / a)
    yendx = int((- c - b * row) / a)
    xendy = int((- c - a * col) / b)
    cv2.circle(image1,(int(x1),int(y1)),5,(0,0,255),-1)
    cv2.circle(image2,(int(x2),int(y2)),5,(0,0,255),-1)
    if 0 < x0y < row:
        if 0 < xendy < row:
            cv2.line(image2,(0,x0y),(col,xendy),(255,255,255),1)
        elif 0 < y0x < col:
            cv2.line(image2,(0,x0y),(y0x,0),(255,255,255),1)
        elif 0 < yendx < col:
            cv2.line(image2,(0,x0y),(yendx,row),(255,255,255),1)
    if 0 < xendy < row:
        if 0 < yendx < col:
            cv2.line(image2,(col,xendy),(yendx,row),(255,255,255),1)
        elif 0 < y0x < col:
            cv2.line(image2,(col,xendy),(y0x,0),(255,255,255),1)
        elif 0 < x0y < row:
            cv2.line(image2,(col,xendy),(0,x0y),(255,255,255),1)
    if 0 < yendx < col:
        if 0 < xendy < row:
            cv2.line(image2,(yendx,row),(col,xendy),(255,255,255),1)
        elif 0 < y0x < col:
            cv2.line(image2,(yendx,row),(y0x,0),(255,255,255),1)
        elif 0 < x0y < row:
            cv2.line(image2,(yendx,row),(0,x0y),(255,255,255),1)
    if 0 < y0x < col:
        if 0 < xendy < row:
            cv2.line(image2,(y0x,0),(col,xendy),(255,255,255),1)
        elif 0 < yendx < col:
            cv2.line(image2,(y0x,0),(yendx,row),(255,255,255),1)
        elif 0 < x0y < row:
            cv2.line(image2,(y0x,0),(0,x0y),(255,255,255),1)
                
combine2 = np.concatenate((image1,image2), axis = 1)
cv2.imwrite('epipolar.png', combine2)   

cv2.imshow('combined_image',combine2)
cv2.waitKey(0)
cv2.destroyAllWindows()