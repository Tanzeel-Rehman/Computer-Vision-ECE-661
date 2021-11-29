# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 17:51:14 2020
@author: Tanzeel Rehman
"""

import cv2
import numpy as np
from scipy import signal as sig


def grayscale(imgname,resize_scale=0.5,resize_falg=True):
    '''
    Read the image and convert it into the grayscale if not already.
    '''
    #Read the color image 
    img_color = cv2.imread(imgname)
    # Adjust the width and height by a constant factor, this maintains the aspect ratio
    if resize_falg == True:
        w = int(img_color.shape[1]*resize_scale)
        h = int(img_color.shape[0]*resize_scale)
        img_color=cv2.resize(img_color,(w,h),interpolation = cv2.INTER_LINEAR)
    else:
        img_color = img_color
    
    #Check if the image has been read
    if img_color is not None:
        #Check if the image is color
        if len(img_color.shape)==3:
            #Convert to gray scale
            img_gray= cv2.cvtColor(img_color,cv2.COLOR_BGR2GRAY)
        elif len(img_color.shape)==1:
            img_gray=img_color
        else:
            print("Image shape not identified")
        return  img_gray,img_color
    else:
        print ("Image not found:"+imgname)
    return None
    

def Haar_kernels(sigma):
    '''
    Get the haar kernels for as a function of sigma.
    The sigma controls the size of kernels. The final output 
    size is smallest even integer > 4*sigma (as per Lecture 9)    
    '''
    kernel_dim=np.ceil(4*sigma)
    if kernel_dim % 2==0:
        kernel_dim=kernel_dim #Scale that rounds up to the even value
    else:
        kernel_dim=kernel_dim+1 #Scale that rounds upto odd values
    kernel_dim=int(kernel_dim)
    # Get the haar kernel in x direction
    kernel_dx=np.concatenate((-1*np.ones((kernel_dim,int(kernel_dim/2))),np.ones((kernel_dim,int(kernel_dim/2)))),axis=1)
    # Get the haar kernel in y direction
    kernel_dy=np.concatenate((np.ones((int(kernel_dim/2),kernel_dim)),-1*np.ones((int(kernel_dim/2),kernel_dim))),axis=0)
    return kernel_dx,kernel_dy

def Get_window(image,kernel_size,x,y):
    '''
    Function for finding the maximum inside a kernel.
    Requires the image, kernelsize and current image coordinates
    to define the current region occupied by kernel
    '''
    # Current kernel centered at x and y
    Window = image[y-kernel_size : y + kernel_size+1, x-kernel_size : x + kernel_size+1]
    #max_val = np.max(Window)
    return Window
            

def Harris_croner(img_gray,k=0.04,sigma=1.2,suppresion_window=14):
    '''
    Function for finding harris corners
    Inputs:
        img_gray: FThe gray image from which the corners will be detected
        k: Cornerness factor, usually 0.04-0.06
        sigma: scale factor at which the corners will be detected
    Output: A list of found corners
    
    '''
    
    #Initialize an empty list for storing corners
    valid_corners = []

    #Calculate the Haar kernel for measuring the gradients in x and y direction
    haar_dx,haar_dy=Haar_kernels(sigma)
    
    #Compute the x and y derivatives by performing the convolution
    #Keep the dimension same to input image
    Ix = sig.convolve2d(img_gray,haar_dx,mode='same')
    Iy = sig.convolve2d(img_gray,haar_dy,mode='same')
    
    #Compute the sqaure and cross gradients for the C matrix
    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix*Iy
    '''
    Calculate the sum of gradient at each pixel by 5*sigma window.
    To avoid 2 foor loops this is done by convolving a filter with equal
    weights =1. Alternatively, Guassian kernel can be used with sigma controlling 
    width to acheive the spatial integration.
    '''
    kernel_dim = 5*sigma
    if kernel_dim % 2==0:
        kernel_dim=kernel_dim+1 #If even make it odd
    else:
        kernel_dim=kernel_dim # else odd
    kernel_dim=np.int(kernel_dim)
    Gauss_kernel = np.ones((kernel_dim,kernel_dim))
    #Get the sum for Ixx, Ixy and Ixy
    Ixx_sum = sig.convolve2d(Ixx,Gauss_kernel,mode='same')
    Iyy_sum = sig.convolve2d(Iyy,Gauss_kernel,mode='same')
    Ixy_sum = sig.convolve2d(Ixy,Gauss_kernel,mode='same')
    
    #Compute the determinant of C matrix, where C = [[Ixx,Ixy],[Ixy,Iyy]]
    detC = (Ixx_sum * Iyy_sum) - (Ixy_sum**2)
    traceC = Ixx_sum + Iyy_sum
    # Compute the cornerness response of Harris detector
    R = detC - k * traceC**2
    #Remove negative entries
    R = R * (R > 0)
    
    # TODO: Vectorize loops --At present time consuming
    # Do max noise suppression to avoid finding the nearby corners 
    #Reference: (https://www.youtube.com/watch?v=_qgKQGsuKeQ&t=2545s,https://muthu.co/harris-corner-detector-implementation-in-python/)
    kernel_half=suppresion_window     #This is half kernel upto which suppression will be done
    for y in range( kernel_half, img_gray.shape[0]-kernel_half):
        for x in range( kernel_half, img_gray.shape[1]-kernel_half):
            Win_img = Get_window(R,kernel_half,x,y)
            if R[ y, x ] == np.max(Win_img):
                valid_corners.append( [x,y] )
    return valid_corners

def Get_SSD_Correspondences(img_1_gray,img_2_gray,Cornerslist_1,Cornerslist_2,window_dim=21,reject_ratio=0.8):
    '''
    Function for getting the corner correspondences from a pair of image using SSD
    Inputs:
        img_1_gray,img_2_gray: First and second gray images
        Cornerslist_1,Cornerslist_2: List of corners for 1st and 2nd image
        window_dim: Window size to define neiighboorhood for computing the Sum of Squared differences
        reject_ratio: Ratio for rejecting the corner. large value (0.8,0.9) = less conservative
                      small value (0.4,0.5) = more conservative
    Output: An array having variable length x 4 contining coordinates of corresponding corners 
    
    '''
    # TODO: Vectorize loops --At present time consuming

    #Convert the corner lists to the arrays
    corner_image1 = np.array(Cornerslist_1)
    corner_image2 = np.array(Cornerslist_2)

    win_half = int(window_dim/2)
    #Initialize an empty list for storing corners
    valid_correspondences = []
    
    #Initialize 2D matrix to store the distances of a specific point in image 1 with everyother point in image 2
    #Size will be num_corners_1 x num_corners_2
    F = np.zeros((len(corner_image1),len(corner_image2)))
    
    #Fill the 2D matrix with SSD distance
    for y in range(len(corner_image1)):
        for x in range(len(corner_image2)):
            f1 =  Get_window(img_1_gray,win_half,corner_image1[y,0],corner_image1[y,1])
            f2 = Get_window(img_2_gray,win_half,corner_image2[x,0],corner_image2[x,1])
            F[y,x] = np.sum((f1-f2)**2)
    
    #Find the valid correspondences via normalized distance and reject if doesn't pass threshold
    for y in range(len(corner_image1)):
        #Find the two minimums for each row of a SSD matrix and normalize the distance
        dual_minima = np.partition(F[y,:],2)[:2]
        #Find the index of 1st minima in every row of SSD_matrix
        x=np.argmin(F[y,:])
        #Check if the normalize distance is less than the predefined threshold, then keep this correspondence otherwise reject it
        if dual_minima[0] / dual_minima[1] < reject_ratio:
            F[:,x] = np.inf   #Mark that this column has been taken to avoid double correspondence (hard learn't lesson)
            valid_correspondences.append([corner_image1[y,0],corner_image1[y,1],corner_image2[x,0],corner_image2[x,1]])
    
        #Find the min and max for every row of the SSD matrix and normalize the distance
        #reject_ratio=1-reject_ratio
        #x=np.argmin(F[y,:])
        #if np.min(F[y,:]) / np.max(F[y,:]) < reject_ratio:
        #    F[:,x] = np.max(F)   #Mark that this column has been taken to avoid double correspondence (hard learn't lesson)
        #    valid_correspondences.append([corner_image1[y,0],corner_image1[y,1],corner_image2[x,0],corner_image2[x,1]])
    
    
    return np.array(valid_correspondences)

def Get_NCC_Correspondences(image1,image2,Cornerslist_1,Cornerslist_2,window_dim=21,reject_ratio=0.8):
    
    #Convert the corner lists to the arrays
    corner_image1 = np.array(Cornerslist_1)
    corner_image2 = np.array(Cornerslist_2)
    
    win_half = int(window_dim/2)
    #Initialize an empty list for storing corners
    valid_correspondences = []
    
    #Initialize 2D matrix to store the distances of a specific point in image 1 with everyother point in image 2
    #Size will be num_corners_1 x num_corners_2
    F = np.zeros((len(corner_image1),len(corner_image2)))
    
    for y in range(len(corner_image1)):
        for x in range(len(corner_image2)):
            f1 =  Get_window(image1,win_half,corner_image1[y,0],corner_image1[y,1])
            f2 = Get_window(image2,win_half,corner_image2[x,0],corner_image2[x,1])
            mean1 = np.mean(f1) 
            mean2 = np.mean(f2)
            numemnator = np.sum((f1- mean1)*(f2- mean2))
            denomenator = np.sqrt((np.sum((f1- mean1)**2))*(np.sum((f2- mean2)**2)))
            F[y,x] = numemnator/denomenator

    #Identify the corresponding corner points in the two images by thresholding 
    for y in range(len(corner_image1)):
        x=np.argmax(F[y,:])
        if F[y,x] > reject_ratio:
            F[:,x] = np.NINF   #Mark that this column has been taken to avoid double correspondence (hard learn't lesson)
            valid_correspondences.append([corner_image1[y,0],corner_image1[y,1],corner_image2[x,0],corner_image2[x,1]])

    return np.array(valid_correspondences)

def draw_and_save(img_color,img_gray,k_ratio,sigma,suppresion_window,savename):
    Cornerslist=Harris_croner(img_gray,k_ratio,sigma,suppresion_window)
    
    for i in range (len(Cornerslist)):
        #Plot a circle of red color with a radius of 3 to mark the corner points.
        cv2.circle(img_color, tuple(Cornerslist[i]), 3, (0,0,255), 2)
        #cv2.imshow(savename,img_color)
        cv2.imwrite(savename,img_color)
    print(len(Cornerslist))
    return Cornerslist 

def Show_Correspondences(img_1_color,img_2_color,corresponding_corners):
    '''
    Function for plotting the corresponding points on the image
    '''
    #Shape of input images
    h1,w1=img_1_color.shape[0:2]
    h2,w2=img_2_color.shape[0:2]
    #Find the maximum height from 2 images. This will be the height of output image
    max_height = max(h1,h2)
    #create an empty image having size of max_height x w1+w2
    plot_img = np.zeros((max_height,(w1+w2),3))
    #Fill empty image with image1 and 2, this will leave empty border on the 
    #image of least height 
    plot_img [0:h1,0:w1,:] = img_1_color
    plot_img [0:h2,w1:,:] = img_2_color
    
    for point in corresponding_corners:
        #Plot a circle of red color with a radius of 3 to mark the corner points on img1.
        cv2.circle(plot_img,tuple(point[0:2]),3,(0,0,255),2)
        #Plot a circle of red color with a radius of 3 to mark the corner points on img2.
        cv2.circle(plot_img,tuple([point[2]+w1,point[3]]),3,(0,0,255),2)  #shift the pointer by width of img1
        #Plot the blue line joining corresponding points on images 
        cv2.line(plot_img,tuple(point[0:2]),tuple([point[2]+w1,point[3]]),(255,0,0),2) 
    return plot_img

 
def Get_SIFT_Features(img_gray,best_features=5000,thresh=0.1):
    #Define the SIFT parameters
    #best_features=5000
    #thresh=0.1 #This will control the contrast impacts
    #Create a SIFT object
    sift = cv2.xfeatures2d.SIFT_create(best_features,contrastThreshold=thresh,sigma=2.6)
    #Compute the sift features and descriptors
    kp, des = sift.detectAndCompute(img_gray,None)
    return kp, des

def Get_BruteForce_Correspondences(sift_des1, sift_des2):
    #Initialize the Brute force matching
    BF_matcher = cv2.BFMatcher()
    
    #Find the 2 valid matches for each point, Use Lowe threshold
    two_matches = BF_matcher.knnMatch(sift_des1,sift_des2,k=2) 
    
    #Store all valid matches
    Valid_matches = [] 
    
    # Pick the best match based on Lowe's paper
    for m1, m2 in two_matches:
        if m1.distance < 0.75 * m2.distance:
            Valid_matches.append([m1])
    
    
    return Valid_matches

def Show_Correspondences_Sift(valid_matches,img_1_color,img_2_color,kp1,kp2):
    #def Show_Correspondences(,img_2_color,corresponding_corners)
    plot_img = cv2.drawMatchesKnn(img_1_color,kp1,img_2_color,kp2,valid_matches,None,flags=2)
    return plot_img

def draw_and_save_Sift(img_color,keypoints,savename):
    output_img=cv2.drawKeypoints(img_color,keypoints,np.array([]),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(savename,output_img)
    

'''--------Main Code for Task 1.1--------''' 

img_1,img_col_1=grayscale("./mypair2/1.JPG",0.75,False)
#Make a copy of color image before plotting
img_col_1_copy=np.copy(img_col_1)
#Pick the different scales,find the corner, plot and save
#draw_and_save(img_col_1,img_1,0.04,0.6,14,'./mypair2_corners/1_Corners_0.6.jpg')
#draw_and_save(np.copy(img_col_1_copy),img_1,0.04,1.2,14,'./mypair2_corners/1_Corners_1.2.jpg')
#draw_and_save(np.copy(img_col_1_copy),img_1,0.04,1.8,16,'./mypair2_corners/1_Corners_1.8.jpg')
draw_and_save(np.copy(img_col_1_copy),img_1,0.04,2.4,18,'./mypair2_corners/1_Corners_2.4.jpg')

#For image 2
img_2,img_col_2=grayscale("./mypair2/2.JPG",0.75,False)
#Make a copy of color image before plotting
img_col_2_copy=np.copy(img_col_2)
#Pick the different scales,find the corner, plot and save
#draw_and_save(img_col_2,img_2,0.04,0.6,14,'./mypair2_corners/2_Corners_0.6.jpg')
#draw_and_save(np.copy(img_col_2_copy),img_2,0.04,1.2,16,'./mypair2_corners/2_Corners_1.2.jpg')
#draw_and_save(np.copy(img_col_2_copy),img_2,0.04,1.8,16,'./mypair2_corners/2_Corners_1.8.jpg')
draw_and_save(np.copy(img_col_2_copy),img_2,0.04,2.4,18,'./mypair2_corners/2_Corners_2.4.jpg')


'''-------Main Code for Task 1.2 (SSD)---------------'''

img_1,img_col_1=grayscale("./mypair2/1.JPG",0.75,False)  #Resize my own images as they are big
img_2,img_col_2=grayscale("./mypair2/2.JPG",0.75,False)

Cornerslist_1=Harris_croner(img_1,0.04,2.4,18)
Cornerslist_2=Harris_croner(img_2,0.04,2.4,18)

SSD_Correspondences = Get_SSD_Correspondences(img_1,img_2,Cornerslist_1,Cornerslist_2,25,0.65)
comb_img_ssd = Show_Correspondences(img_col_1,img_col_2,SSD_Correspondences)
cv2.imwrite('./mypair2_corners/SSD_Correspondences_2.4.jpg',comb_img_ssd)

'''-------Main Code for Task 1.3 (NCC)---------------'''

img_1,img_col_1=grayscale("./mypair2/1.JPG",0.75,False)  #Resize my own images as they are big
img_2,img_col_2=grayscale("./mypair2/2.JPG",0.75,False)

#Cornerslist_1=Harris_croner(img_1,0.06,0.6,21)
#Cornerslist_2=Harris_croner(img_2,0.06,0.6,21)

NCC_Correspondences = Get_NCC_Correspondences(img_1,img_2,Cornerslist_1,Cornerslist_2,25,0.9)
comb_img_ncc = Show_Correspondences(img_col_1,img_col_2,NCC_Correspondences)
cv2.imwrite('./mypair2_corners/NCC_Correspondences_2.4.jpg',comb_img_ncc)

    
'''-------Main Code for Task 1.4 (SIFT)-----------'''
'''
#Read the images
img_1,img_col_1=grayscale("./mypair1/1.JPG",0.5,True)  #Resize my own images as they are big
img_2,img_col_2=grayscale("./mypair1/2.JPG",0.5,True)  #Resize my own images as they are big
#Get the keypoints
kp1,des1=Get_SIFT_Features(img_1)
kp2,des2=Get_SIFT_Features(img_2)
#Draw and save the output images
draw_and_save_Sift(img_col_1,kp1,'./mypair1_corners/1_SIFT.jpg')
draw_and_save_Sift(img_col_1,kp2,'./mypair1_corners/1_SIFT.jpg')


# Find the correspondences, draw them and save the final image
SIFT_Correspondences = Get_BruteForce_Correspondences(des1,des2)
comb_img_sift = Show_Correspondences_Sift(SIFT_Correspondences,img_col_1,img_col_2,kp1,kp2)
cv2.imwrite('./mypair1_corners/SIFT_Correspondences.jpg',comb_img_sift)
'''