# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:59:51 2020

@author: Tanzeel Rehman

TODO: Vectorize the loops for better resource utilization

"""

import numpy as np
import cv2

class Homography(object):
    
    def __init__(self,Range_img):
        """
        Class Responsible for computing and applying Homograpies.
        Inputs:
            Domain_pts: An n x 2 array containing coordinates of domian image points(Xi,Yi)
            range_point: An n x 2 array containing coordinates of range image points(Xi',Yi')
            Domain_img: Domain image
            Range_img: Range image
       
        """
        #self.Domain_pts=Domain_pts
        #self.Range_pts=Range_pts
        #self.Domain_img=Domain_img
        self.Range_img=Range_img
    def find_homography (self,Domain_pts,Range_pts):
        '''
        function for estimating the 3 x 3 Homography matrix 
            Output: A 3 x 3 Homography matrix 
        '''
        # Find num of points provided
        n = Domain_pts.shape[0]
        #Initialize A Design matrix having size of 2n x 8
        A = np.zeros((2*n,8))
        #Reshape the Range_pts to 1D vector of size 2*num_pts x 8
        y = Range_pts.reshape((2*n, 1))
        #Loop through all the points provided and stack them vertically, this will result in 2n x 8 Design matrix
        for i in range (n):
            A[i*2:i*2+2]=self.Get_A_matrix(Domain_pts[i],Range_pts[i])
        '''
        Compute the h vector (2n x 1) by using least sqaures solution. In case we have unique
        solution from 4 points, we can directly inverse the A design matrix. But for the overdetermined
        system, we can use the least sqaures estimation.
        '''
        h=np.dot(np.linalg.inv(np.dot(A.T,A)),np.dot(A.T,y))
        # Add the last element as 1 in the h vector to obtain HC. Now h will be 2*n+1 x 1 vector.
        h = np.concatenate((h,1), axis=None)
        #Reshape the h vector to H homography matrix
        H = h.reshape((3,3))
        return H


    def Get_A_matrix(self,domain_point,range_point):
        '''
        function for generating a 2 x 8 design matrix needed to compute Homography
        Inputs:
            domain_point: Coordinates of a point in the domain image (x,y)
            range_point: Coordinates of corresponding point in the range image (x',y')
        Output: A 2 x 8 design matrix 
        '''
        #Extract the x and y coordinates from a point pair
        x,y=domain_point[0], domain_point[1]
        xr,yr=range_point[0], range_point[1]
        #  Make A matrix
        A=np.array([[x,y,1,0,0,0,-x*xr,-y*xr],[0,0,0,x,y,1,-x*yr,-y*yr]])
        return A

    def transform_image(self,Domain_img,Domain_pts,Homography,task):
        '''
        function for applying the 3 x 3 Homography to the domian image to obtain new range pixel 
        coordinates and then painting the range image onto the domian image using the new range pixel coords. 
        range 
            Output: A new domian image with the range image projected on it.
        '''
        #Get the height and width of Domain Image
        height, width = Domain_img.shape[:2]
        if task==1:
            Domain_img_gen = Domain_img
        elif task==2:
            Domain_img_gen = np.zeros (Domain_img.shape,dtype='uint8')
        else:
            raise NotImplementedError('Task not implemented. Pick 1 or 2 for 1.1 and 1.2, respectively')
        
        # Mask the PQRS region of the Domain image using fillpolygon tool, since it is not rectangle
        masked_domain = cv2.fillPoly(np.zeros(Domain_img.shape[0:2],dtype='uint8'),[Domain_pts],255)    
        # Traverse through each Domain image pixel by pixel   TODO: Vectorize the loops
        # Loop for controlling the columns of image
        for x in range(width):
            #loop for controlling the rows of image
            for y in range(height):
                #Check if we are inside the polygon defined by Domain_pts
                if (masked_domain[y,x]) > 0:
                    #Convert the domain pixel coordinates from the physical space to HC
                    Pix_domain = np.array([x,y,1])
                    #Apply the homogrpahy to the HC pixel coordinates to obtain the Range coordinates X'=HX
                    Pix_range=Homography@ Pix_domain
                    #Convert from the range image pixel HC to physical space 
                    Pix_range =Pix_range/Pix_range[2]
                    # Find the nearest neighbor by rounding to the nearest integer as pix coordinates are ints
                    Pix_range = np.round(Pix_range).astype(np.int)
                    #Check if the new range coords computed from Homography are within bounds of actual range image 
                    #if(Pix_range[0] > 0 & Pix_range[1] > 0 & Pix_range[0] < self.Range_img.shape[1] & Pix_range[1] < self.Range_img.shape[0]):
                    if(Pix_range[0] < self.Range_img.shape[1]) & (Pix_range[1] < self.Range_img.shape[0]):
                        #Paint the Range image on the Domain image using new range coords
                        Domain_img_gen[y,x] = self.Range_img[Pix_range[1]][Pix_range[0]]
        return Domain_img_gen
    

'''-------Main Code for Task-1.1---------'''
# Sequentially go over through 3 painting images and project the kittens

#Function for calling the Homography class and saving results 
def Save(Domain_img,Range_image,Domain_pts,Range_pts,savefileint,strfile):
    # Initialize and call to the class responsible for computing and applying Homography
    homo=Homography(Range_image)
    H=homo.find_homography(Domain_pts,Range_pts)
    img_painted1=homo.transform_image(Domain_img,Domain_pts,H,1)
    savefilename=strfile+str(savefileint)+"Projected.jpeg"
    cv2.imwrite(savefilename,img_painted1)


# Read all the painitng images (Domain Images) 
painting_1 = cv2.imread('painting1.jpeg')
painting_2 = cv2.imread('painting2.jpeg')
painting_3 = cv2.imread('painting3.jpeg')
# Read the kitten image (Range Image)
image_kittens=cv2.imread('kittens.jpeg')

# Make 2D array of corner points of rectangles in different painitng images 
PSRQ_Painting1 = np.array([[233,412],[1925,202],[1820,1988],[139,1693]])
PSRQ_Painting2 = np.array([[166,515],[1966,631],[1969,2096],[172,2525]])
PSRQ_Painting3 = np.array([[74,356],[1366,110],[1204,2064],[67,1422]])
# Make a 2D array of corner points of Kittens
PSRQ_kits = np.array([[0,0],[image_kittens.shape[1]-1,0],[image_kittens.shape[ 1]-1,
                       image_kittens.shape[0]-1],[0,image_kittens.shape[0 ]-1 ]])
  
#Save the results for task 1.1
Save(painting_1,image_kittens,PSRQ_Painting1,PSRQ_kits,1,"painting1")
Save(painting_2,image_kittens,PSRQ_Painting2,PSRQ_kits,2,"painting2")
Save(painting_3,image_kittens,PSRQ_Painting3,PSRQ_kits,3,"painting3")

'''-------Main Code for Task-1.2---------'''
painting_1 = cv2.imread('painting1.jpeg')
painting_2 = cv2.imread('painting2.jpeg')
painting_3 = cv2.imread('painting3.jpeg')

homo=Homography(painting_1)
H12= homo.find_homography (PSRQ_Painting2 , PSRQ_Painting1)
H23= homo.find_homography (PSRQ_Painting3 , PSRQ_Painting2)
H = np.dot(H12,H23)
Domian_pts_21= np.array([[0,0],[painting_3.shape[1]-1,0],[painting_3.shape[1]-1,painting_3.shape[0]-1],[0,painting_3.shape[0 ]-1]])
img_painted1=homo.transform_image(painting_3,Domian_pts_21,H,2)
cv2.imwrite("Painting1_to_3.jpeg",img_painted1)


'''-------Main Code for Task-2.1---------'''
# Read all the painitng images (Domain Images) 
painting_1 = cv2.imread('My_1a.jpg')
painting_2 = cv2.imread('My_1b.jpg')
painting_3 = cv2.imread('My_1c.jpg')
# Read the kitten image (Range Image)
image_kittens=cv2.imread('copy.jpg')

# Make 2D array of corner points of rectangles in different painitng images 
PSRQ_My1 = np.array([[396,654],[3841,250],[3563,5856],[876,4819]])
PSRQ_My2 = np.array([[628,242],[3416,922],[2998,4464],[602,5315]])
PSRQ_My3 = np.array([[224,599],[2630,306],[2562,5554],[661,4198]])
# Make a 2D array of corner points of Kittens
PSRQ_Note = np.array([[0,0],[image_kittens.shape[1]-1,0],[image_kittens.shape[ 1]-1,
                       image_kittens.shape[0]-1],[0,image_kittens.shape[0 ]-1 ]])

#Save the results for task 2.1
Save(painting_1,image_kittens,PSRQ_My1,PSRQ_Note,1,"My_file")
Save(painting_2,image_kittens,PSRQ_My2,PSRQ_Note,2,"My_file")
Save(painting_3,image_kittens,PSRQ_My3,PSRQ_Note,3,"My_file")


'''-------Main Code for Task-2.2---------'''
painting_1 = cv2.imread('My_1a.jpg')
painting_2 = cv2.imread('My_1b.jpg')
painting_3 = cv2.imread('My_1c.jpg')

homo=Homography(painting_1)
H12= homo.find_homography (PSRQ_My2 , PSRQ_My1)
H23= homo.find_homography (PSRQ_My3 , PSRQ_My2)
H = np.dot(H12,H23)
Domian_pts_21= np.array([[0,0],[painting_3.shape[1]-1,0],[painting_3.shape[1]-1,painting_3.shape[0]-1],[0,painting_3.shape[0 ]-1]])
img_painted1=homo.transform_image(painting_3,Domian_pts_21,H,2)
cv2.imwrite("MY_1_to_3.jpeg",img_painted1)