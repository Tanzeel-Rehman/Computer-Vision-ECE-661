# -*- coding: utf-8 -*-
"""
Created on Sat Oct 08 13:41:47 2020

@author: Tanzeel
"""

import cv2
import numpy as np

def Otsu_GrayScale(Gray_image, channel_mask=None):
    '''
    Function for obtaining a segmentation mask using grayscale image
    Inputs:
        Gray_image: A single channel grayscale image
        channel_mask: An existing or no mask for itetratively refining the segmentation results
    Output: A signle channel Segmentation mask
    '''
    #Initialize the parameters
    distribution = np.zeros((256,1))
    max_sigma_b_squared =-1;
    Thresholded_img = np.copy(Gray_image)
    #probabilities and mean of levels lower than gray value
    omega0 = 0
    mu = 0

    #For the 1st iteration mask is none, so use original image parameters
    if channel_mask is None:
        histogram, _ = np.histogram(Gray_image, bins=256, range=(0, 255))
        mu_total = np.mean(Gray_image)
    else:
        '''
        This is 2nd to onwards iterations. Perform elementwise multiplication
        to get rid of background using mask from previous iteration and compute
        mean and hsitogram of only non-zero entries of image
        '''
        masked_img=np.multiply(Gray_image,(channel_mask/255))
        #Remove the zeros from histogram as they are coming from existing background
        histogram, _ = np.histogram(masked_img, bins=256, range=(0.5, 255.5))
        #Set zro entries to nan
        masked_img[masked_img==0]=np.nan
        mu_total=np.nanmean(masked_img)
       
    #Distribution from histogram
    distribution = histogram / np.sum(histogram)
    
    for k in range(256): 
        omega0 = omega0 + distribution[k]
        omega1 = 1 - omega0
        mu = mu + k*distribution[k]
        #Avoid dividing by zero warning
        if omega0 == 0 or omega1==0:
            continue
        #Between class variance
        sigma_b_squared = ((mu_total * omega0 - mu)**2)/(omega0*omega1)
        #Find the new threshold level based on the between class variance
        if sigma_b_squared > max_sigma_b_squared:
            max_sigma_b_squared = sigma_b_squared
            otsu_threshold = k
    print("Best Otsu Threshold found:",otsu_threshold)
    #Check if threshold is greater than zero else generate an empty image
    if otsu_threshold > 0 :
        Thresholded_img[Thresholded_img>otsu_threshold]=255
        Thresholded_img[Thresholded_img<otsu_threshold]=0  
    else:          
        Thresholded_img=np.zeros_like(Gray_image)
   
    return Thresholded_img

def Otsu_RGB(Color_img, savename, itertions = [1,1,1]):
    '''
    Function for obtaining a composite segmentation mask using RGB image
    Inputs:
        Color_img: A three channel RGB image
        savename: A string for saving the channel wise segmentation mask
        iterations: A list indicating the number of iterations for every channel
    Output: A composite Segmentation mask
    '''
    # Get the shape of image
    h,w,channels=Color_img.shape
    # Array for saving the masks for all channels
    masks=np.zeros_like(Color_img)
    #Composite RGB mask
    RGB_mask = np.zeros((h,w),np.uint8)
        
    # Pass through all image channels and extract the binary mask
    for channel in range(channels):
        '''
        Flag for the 1st iteration, in the next iterations the mask from previous
        iteration will be used
        '''
        channel_wise_mask = None        
        # Run the Otsu algorithm iteratively
        for n in range(itertions[channel]):
            img_gray = Color_img[:,:,channel]
            channel_wise_mask = Otsu_GrayScale(img_gray,channel_wise_mask)
            #Save the channelwise mask            
            savefilename=savename+ '_Ch_' + str(channel) + '_iter_' +str(n+1) + '.jpg'
            cv2.imwrite(savefilename,channel_wise_mask)
        #Arrange the masks from different channels in an array
        masks[:,:,channel] = channel_wise_mask
        #Perform And to obtain the final RGB mask
        RGB_mask = masks[:,:,0] & masks[:,:,1] & masks[:,:,2]
        #out_img=np.bitwise_and(RGB_mask, mask_ch)
    return RGB_mask

def Texture_Image(Gray_image, kernel_sizes):
    # Get the shape of image
    h,w=Gray_image.shape
    
    # Generate an empty variance matrix with masks from different windows being concatenated as 3rd channel
    variances = np.zeros((h,w,len(kernel_sizes)),np.uint8)
    #Loop through all the windows given by a list
    for k, ksize in enumerate(kernel_sizes):
        #Get the slidding window and compute its texture statistics
        kernel_half = np.int(ksize/2)
        #TODO: Change the loops to vectorized version. Time consuming
        for y in range(kernel_half, h - kernel_half):
            for x in range(kernel_half, w- kernel_half):
                #Window = Gray_image[y-kernel_half : y + kernel_half+1, x-kernel_half : x + kernel_half+1]
                #vari = np.var(Window)
                variances[y,x,k] = Get_window(Gray_image,kernel_half,x,y)
    return variances
#The contour extraction algorithm. This is done after segmentation  
def getForegroundContour(img):
    
    #Initialize final contour image
    out_img = np.zeros((img.shape[0],img.shape[1]),dtype='uint8')
    
    #Mark border if pixel value is not 0 AND one of 8-neighbor is 0
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            if img[i,j]!=0 and np.min(img[i-1:i+2,j-1:j+2])==0:
                out_img[i,j] = 255
            
    return out_img

def Get_Contours(masked_image):
    # Get the shape of image
    h,w=masked_image.shape
    #Generate an empty image to hold the contours
    contour_img = np.zeros(masked_image.shape,dtype=np.uint8)
    for i in range(1,h-1):
        for j in range(1,w-1):
            if masked_image[i,j]==255:  #Check if we are in foreground region
                #Check neighboorhood
                window=masked_image[i-1:i+2,j-1:j+2]
                #Check if all are foreground pixels or not
                if np.sum(window) < 255*9:
                    contour_img[i,j]=255
           
    return contour_img

def Get_window(image,kernel_size,x,y):
    '''
    Function for finding thevariance inside a kernel.
    Requires the image, kernelsize and current image coordinates
    to define the current region occupied by kernel
    '''
    # Current kernel centered at x and y
    Window = image[y-kernel_size : y + kernel_size+1, x-kernel_size : x + kernel_size+1]
    vari = np.var(Window)
    return np.int(vari)

def erode_mask(mask,errode_size):
    struct = np.ones((errode_size,errode_size),np.uint8)
    mask = cv2.erode(mask,struct)
    return mask
def dilate_mask(mask,dilate_size):
    struct = np.ones((dilate_size,dilate_size),np.uint8)
    mask = cv2.dilate(mask,struct)
    return mask

'''----------------Main Code for Task 1.1-------------'''
im1=cv2.imread('cat.jpg')
iters_channel_1 = [1,1,2]
Segmented_1 = Otsu_RGB(im1,'cat',iters_channel_1)
cv2.imwrite('cat' +  '_segmented.jpg',Segmented_1)
#Perform morphological operations and get contours
Segmented_1 = erode_mask(Segmented_1,2)
Segmented_1 = dilate_mask(Segmented_1,2)
Contour_Seg_1 = Get_Contours(Segmented_1)
cv2.imwrite('cat' +  '_Contours.jpg',Contour_Seg_1)
print('-------------------------------')

im2=cv2.imread('pigeon.jpg')
iters_channel_2 = [1,1,2]
Segmented_2 = Otsu_RGB(im2,'pigeon',iters_channel_2)
cv2.imwrite('pigeon' +  '_segmented.jpg',Segmented_2)
#Perform morphological operations and get contours
Segmented_2 = erode_mask(Segmented_2,2)
Segmented_2 = dilate_mask(Segmented_2,2)
Contour_Seg_2 = Get_Contours(Segmented_2)
cv2.imwrite('pigeon' +  '_Contours.jpg',Contour_Seg_2)
print('-------------------------------')

im3=cv2.imread('Red_Fox_.jpg')
iters_channel_3 = [1,1,2]
Segmented_3 = Otsu_RGB(im3,'Red_Fox_',iters_channel_3)
cv2.imwrite('Red_Fox_' +  '_segmented.jpg',Segmented_3)
#Perform morphological operations and get contours
Segmented_3 = erode_mask(Segmented_3,4)
Segmented_3 = dilate_mask(Segmented_3,2)
Contour_Seg_3 = Get_Contours(Segmented_3)
cv2.imwrite('pigeon' +  '_Contours.jpg',Contour_Seg_3)
print('-------------------------------')

'''------------Main Code for Task 1.2---------------'''
image_gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
window_size = [3,5,7]
text_img_1 = Texture_Image(image_gray1,window_size)
Segmented_Text_1 = Otsu_RGB(text_img_1,'cat_Texture',iters_channel_1)
cv2.imwrite('cat_Texture' +  '_segmented.jpg',Segmented_Text_1)
#Perform morphological operations and get contours
Segmented_Text_1 = dilate_mask(Segmented_Text_1,5)
Segmented_Text_1 = erode_mask(Segmented_Text_1,2)
Contour_Tex_1 = Get_Contours(Segmented_Text_1)
cv2.imwrite('cat_Texture' +  '_Contours.jpg',Contour_Tex_1)
print('-------------------------------')

image_gray_2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
text_img_2 = Texture_Image(image_gray_2,window_size)
Segmented_Text_2 = Otsu_RGB(text_img_2,'pigeon_Texture',[1,1,1])
cv2.imwrite('pigeon_Texture' +  '_segmented.jpg',Segmented_Text_2)
#Perform morphological operations and get contours
Segmented_Text_2 = dilate_mask(Segmented_Text_2,2)
Segmented_Text_2 = erode_mask(Segmented_Text_2,2)
Contour_Tex_2 = Get_Contours(Segmented_Text_2)
cv2.imwrite('pigeon_Texture' +  '_Contours.jpg',Contour_Tex_2)
print('-------------------------------')

image_gray_3 = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)
text_img_3 = Texture_Image(image_gray_3,window_size)
Segmented_Text_3 = Otsu_RGB(text_img_3,'Red_Fox_Texture',[1,1,1])
cv2.imwrite('Red_Fox_Texture' +  '_segmented.jpg',Segmented_Text_3)
#Perform morphological operations and get contours
Segmented_Text_3 = dilate_mask(Segmented_Text_3,2)
Segmented_Text_3 = erode_mask(Segmented_Text_3,2)
Contour_Tex_3 = Get_Contours(Segmented_Text_3)
cv2.imwrite('Red_Fox_Texture' +  '_Contours.jpg',Contour_Tex_3)
print('-------------------------------')