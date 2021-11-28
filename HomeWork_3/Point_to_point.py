# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 20:44:00 2020

@author: Tanzeel
"""

import numpy as np
import cv2

def find_homography (Domain_pts, Range_pts):
    '''
    function for estimating the 3 x 3 Homography matrix 
     Inputs:
        Domain_pts: An n x 2 array containing coordinates of domian image points(Xi,Yi)
        range_point: An n x 2 array containing coordinates of range image points(Xi',Yi')
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
        A[i*2:i*2+2]=Get_A_matrix(Domain_pts[i],Range_pts[i])
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


def Get_A_matrix(domain_point,range_point):
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
       
def find_VL_homography (Domain_pts):
    
    # PQ line is parallel to RS (First Pair)
    # PQ Line
    PQ = np.cross(np.append(Domain_pts[0],1),np.append(Domain_pts[1],1))
    # RS Line
    RS = np.cross(np.append(Domain_pts[3],1),np.append(Domain_pts[2],1))
    
    # First Vanishing point
    Vanish_1 = np.cross(PQ,RS)
    #From HC to physical space
    Vanish_1 = Vanish_1/Vanish_1[2]
    
    # PR line is parallel to QS
    PR = np.cross(np.append(Domain_pts[0],1),np.append(Domain_pts[3],1))  
    QS = np.cross(np.append(Domain_pts[1],1),np.append(Domain_pts[2],1))
    
    # First Vanishing point
    Vanish_2 = np.cross(PR,QS)
    #From HC to physical space
    Vanish_2 = Vanish_2/Vanish_2[2]
    
    # Vanishing line from Vanishing points
    Van_line = np.cross(Vanish_1,Vanish_2)
    #HC to Physical
    Van_line = Van_line/Van_line[2]
    
    #Compute the Homography matrix
    H_VL = np.identity(3)
    H_VL[2,:] = Van_line
    
    return H_VL

def find_Aff_homography(DomPts_ProjectCorr):
    '''
    Function for correcting the affine distortion in the images using 
    real-world orthogonal lines. This function needs to be used for computing 
    the homography after correctiing the projective transformation.
    '''
    # Initialize the A and S matrices having 2 x 2 size.
    A = np.zeros((2,2))
    S_mat = np.ones((2,2))
    # PQ line is perpendicular to QS (First Pair)
    # PQ Line
    PQ_l1 = np.cross(np.append(DomPts_ProjectCorr[0],1),np.append(DomPts_ProjectCorr[1],1))
    PQ_l1 = PQ_l1/PQ_l1[2]  #HC to physical
    # QS Line
    QS_m1 = np.cross(np.append(DomPts_ProjectCorr[1],1),np.append(DomPts_ProjectCorr[2],1))
    QS_m1 = QS_m1/QS_m1[2]
    
    # PS line is perpendicular to QR (diagonal pair)
    PS_l2 = np.cross(np.append(DomPts_ProjectCorr[0],1),np.append(DomPts_ProjectCorr[2],1))  
    PS_l2 = PS_l2/PS_l2[2]
    QR_m2 = np.cross(np.append(DomPts_ProjectCorr[1],1),np.append(DomPts_ProjectCorr[3],1))
    QR_m2 = QR_m2/QR_m2[2]
    
    #Create A and B matrices
    A[0,:] = [PQ_l1[0]*QS_m1[0], PQ_l1[0]*QS_m1[1] + PQ_l1[1]*QS_m1[0]]
    A[1,:] = [PS_l2[0]*QR_m2[0], PS_l2[0]*QR_m2[1] + PS_l2[1]*QR_m2[0]]
    
    Y = np.array([-PQ_l1[1]*QS_m1[1], -PS_l2[1]*QR_m2[1]])
    # Computing the S matrix
    S = np.linalg.inv(A)@Y  #No need of least squares as A is 2 x 2
    S_mat[0,:] = [S[0], S[1]]
    S_mat[1,0] = S[1]
    # Get the matrix A
    h=Decompose_S_mat(S_mat)
    #Compute the Homography matrix
    H_Aff = np.identity(3)
    H_Aff[0:2,0:2] = h
    return H_Aff


def find_homography_one(Domain_pts):
    '''
    Find the homography using 5 pairs of orthogonal lines (Theta=90) using one-step method
    '''
    #Initialize A Design matrix having size of 5 x 5 and S matrix of size 2x2 
    A=np.zeros((5,5))
    Y=np.zeros((5,1))
    S_mat = np.zeros((2,2))
    
    # PQ line is perpendicular to PR (First Pair)
    PQ_l1 = np.cross(np.append(Domain_pts[0],1),np.append(Domain_pts[1],1))
    PR_m1 = np.cross(np.append(Domain_pts[0],1),np.append(Domain_pts[3],1))
    PQ_l1 = PQ_l1/PQ_l1[2]  #HC to physical
    PR_m1 = PR_m1/PR_m1[2]

    # PR line is perpendicular to RS (2nd Pair)
    PR_l2 = np.cross(np.append(Domain_pts[0],1),np.append(Domain_pts[3],1))
    RS_m2 = np.cross(np.append(Domain_pts[3],1),np.append(Domain_pts[2],1))
    PR_l2 = PR_l2/PR_l2[2]  #HC to physical
    RS_m2 = RS_m2/RS_m2[2]
    
    # QS line is perpendicular to SR (3rd Pair)
    QS_l3 = np.cross(np.append(Domain_pts[1],1),np.append(Domain_pts[2],1))
    SR_m3 = np.cross(np.append(Domain_pts[2],1),np.append(Domain_pts[3],1))
    QS_l3 = QS_l3/QS_l3[2]  #HC to physical
    SR_m3 = SR_m3/SR_m3[2]
    
    # PQ line is perpendicular to QS (4th Pair) 
    PQ_l4 = np.cross(np.append(Domain_pts[0],1),np.append(Domain_pts[1],1))
    QS_m4 = np.cross(np.append(Domain_pts[1],1),np.append(Domain_pts[2],1))
    PQ_l4 = PQ_l4/PQ_l4[2]  #HC to physical
    QS_m4 = QS_m4/QS_m4[2]
    
    # PS line is perpendicular to RQ (5th Pair) 
    PS_l5 = np.cross(np.append(Domain_pts[0],1),np.append(Domain_pts[2],1))
    QR_m5 = np.cross(np.append(Domain_pts[1],1),np.append(Domain_pts[3],1))
    PS_l5 = PS_l5/PS_l5[2]  #HC to physical
    QR_m5 = QR_m5/QR_m5[2]
    # Get the A and Y matrices for each pair of line
    A[0,:],Y[0] = Get_A_Y_vectors_one(PQ_l1,PR_m1)
    A[1,:],Y[1] = Get_A_Y_vectors_one(PR_l2,RS_m2)
    A[2,:],Y[2] = Get_A_Y_vectors_one(QS_l3,SR_m3)
    A[3,:],Y[3] = Get_A_Y_vectors_one(PQ_l4,QS_m4)
    A[4,:],Y[4] = Get_A_Y_vectors_one(PS_l5,QR_m5)
    
    #Calculate and normalze C @ inf matrix
    c_inf=np.linalg.inv(A)@Y
    c_inf = c_inf/np.max(c_inf)
    # Computing the S matrix
    S_mat[0,:] = [c_inf[0],c_inf[1]/2]
    S_mat[1,:] = [c_inf[1]/2,c_inf[2]]
    # Get the matrix A
    h=Decompose_S_mat(S_mat)
    # Computer the vector v
    v = (np.linalg.inv(h))@(np.array([c_inf[3]/2,c_inf[4]/2]))
   #Compute the Homography matrix
    H_one = np.zeros((3,3))
    H_one[0]=np.append(h[0,:], 0)
    H_one[1]=np.append(h[1,:], 0)
    H_one[2]=np.append(v, 1)
    
    return H_one

def Decompose_S_mat(S_mat):
    #SVD decomposition of S matrix
    U, D, V = np.linalg.svd(S_mat, full_matrices=True)
    #Diagonlize the D matrix
    Ss=np.diag(D)
    D=np.sqrt(Ss)   #Sqaure root
    # Matrix A
    h = (V@D)@V.T
    return h

def Get_A_Y_vectors_one(l,m):
    '''
    function for generating a 5 x 1 design matrix (A) and a Y vector 
    needed to compute Homography
    Inputs:
        Two orthogonal lines
    Output: A 5 x 1 design matrix and 5x 1 Y vector 
    '''
    # Extract the coordinates of two lines
    l1,l2,l3=l[0], l[1],l[2]
    m1,m2,m3=m[0], m[1],m[2]
    # Make A matrix
    A = np.array([l1*m1, 0.5*(l2*m1 + l1*m2), l2*m2, 0.5*(l1*m3 + l3*m1), 0.5*(l3*m2 + l2*m3)])
    Y = -l3*m3
    return A,Y

def ProjectiveCorr_Points(H_VanishLine,DomPts):
    '''
    Function for applying the homography calculated using the vanishing line method on the
    image coordinates picked for removing the affine distortion. Then supply these projective 
    corrected coordinates to the function "find_Aff_homography"
    
    '''
    #Apply the projective homograpy
    P_pr_corr = H_VanishLine@np.append(DomPts[0],1)
    Q_pr_corr = H_VanishLine@np.append(DomPts[1],1)
    S_pr_corr = H_VanishLine@np.append(DomPts[2],1)
    R_pr_corr = H_VanishLine@np.append(DomPts[3],1)
    
    #Convert from HC to Physical space
    P_pr_corr = np.floor(P_pr_corr/P_pr_corr[2])
    Q_pr_corr = np.floor(Q_pr_corr/Q_pr_corr[2])
    R_pr_corr = np.floor(R_pr_corr/R_pr_corr[2])
    S_pr_corr = np.floor(S_pr_corr/S_pr_corr[2])

    
    dom_pts_corr=np.array([P_pr_corr[0:2],Q_pr_corr[0:2],S_pr_corr[0:2],R_pr_corr[0:2]])
    # Get the affine
    H_Aff = find_Aff_homography(dom_pts_corr)
    return H_Aff

def Bounds_Undistorted(Homography,image):
    #Shape of the distorted image
    image_shape = image.shape

    #Distorted Homogeneous Coordinates of image Bounds
    ImgP= np.array([0,0,1])     # Top left corner of image (X,Y,1)
    ImgQ= np.array([image_shape[1],0,1])    # Top right corner 
    ImgS = np.array([image_shape[1],image_shape[0],1]) #Bottom right
    ImgR = np.array([0,image_shape[0],1])   #bottom left
    
    #Apply the homography on the distroted image bounds to obtain the Corrected image bounds
    WorldP = np.dot(Homography,ImgP)
    WorldP = WorldP/WorldP[2]
    WorldQ = np.dot(Homography,ImgQ)
    WorldQ = WorldQ/WorldQ[2]
    WorldS = np.dot(Homography,ImgS)
    WorldS = WorldS/WorldS[2]
    WorldR = np.dot(Homography,ImgR)
    WorldR = WorldR/WorldR[2]
    
    #Find the extreme points of the corrected image bounds 
    max_point = np.maximum(np.maximum(np.maximum(WorldP ,WorldQ ), WorldS), WorldR)
    min_point = np.minimum (np.minimum (np.minimum(WorldP,WorldQ), WorldS), WorldR)
    #Find the coordinates of cextreme points of corrected image bounds 
    xmax,ymax = max_point[0],max_point[1]
    xmin,ymin = min_point[0],min_point[1]
    # New width and height of corrected image
    width_corr = xmax-xmin
    height_corr =ymax-ymin
    #Compute the GSD in x and y direction. This should be roughly equal to the (width_pixels/physical_width) 
    scalex = image_shape[1]/width_corr
    scaley = image_shape[0] / height_corr
    
    #Scale the new image using max scale
    scale=max(scalex,scaley)
    
    #Create an empty corrected image that will be further filled
    corrected_image = np.zeros((int(np.round(height_corr*scale)), int(np.round(width_corr*scale)), 3),dtype='uint8')
    
    return xmin, ymin, scale, corrected_image

def get_grid(x, y):
    #Get the array of coordinates of image pixels 
    pix_coords = np.indices((x, y))
    # Flatten coordinates into 2*width*height
    pix_coords = pix_coords.reshape(2, -1)
    # Make the pixel coordinates to HC  
    pix_coords=np.append(pix_coords,np.full((1,pix_coords.shape[1]),1),axis=0)
    return pix_coords

def transform_image(xmin, ymin, scaleFactor, undis_image,H_inv,image_org):
        '''
        Function responsible for the transforming the image
        This is the vectorized implementation of HW2 code
        '''
        #Get the height and width of undistorted empty Image
        height, width = undis_image.shape[:2]
        # Get the coordinates         
        coords = get_grid(width, height)
        x_domain, y_domain = coords[0], coords[1]
        # Scale while warping + add the min limits to coordinates of undistorted empty image
        coords = coords.astype(np.int)
        
        coords[0]=(coords[0]/scaleFactor) + xmin
        coords[1]=(coords[1]/scaleFactor) + ymin
        #Apply the inverse Homography to the coordinates
        Pix_range = H_inv@ coords
        #Convert from the range image pixel HC to physical space 
        Pix_range =Pix_range/Pix_range[2]
        Pix_range = np.floor(Pix_range).astype(np.int)
        # Coordinates of range image     
        xcoord_range, ycoord_range = Pix_range[0, :], Pix_range[1, :]
        
        #print(min(x_ori),max(x_ori))
        #print(min(xcoord2),max(xcoord2))
        
        # Find the pixels of distroted image that are withinbounds
        indices = np.where((xcoord_range >= 0) & (xcoord_range < image_org.shape[1]) &
                   (ycoord_range >= 0) & (ycoord_range < image_org.shape[0]))
        # pixel coordinates on both domain and range image which are within bounds
        xpix, ypix = x_domain[indices], y_domain[indices]
        xpix2, ypix2 = xcoord_range[indices], ycoord_range[indices]
        # Map the image
        undis_image[ypix, xpix] = image_org[ypix2,xpix2]
        return undis_image
        #return (xpix,ypix,xpix2,ypix2)
def getImage(offsetX, offsetY, scaleFactor, world_image,H_inv_,image_org):
    for i in range(0,world_image.shape[1]-1): #X-cooridnate, col
        for j in range(0,world_image.shape[0]-1): #Y-coordinate, row
            k1 = i/scaleFactor + offsetX
            k2 = j/scaleFactor + offsetY
            X_domain = [k1,k2]
            X_domain = np.array(X_domain)
            X_domain = np.append(X_domain,1)
            X_range = np.matmul(H_inv_, X_domain)
            X_range = X_range/X_range[-1]
            X_range = np.rint(X_range)
            X_range = X_range.astype(int)
            if(X_range[0] > 0 and X_range[1] > 0 and X_range[0] < image_org.shape[1]-1 and X_range[1] < image_org.shape[0]-1):
                #world_image[j][i] = WeightedAverageRGBPixelValue(X_range,image_org)
                world_image[j][i] = image_org[X_range[1]][X_range[0]]
                
    return world_image


"""

'''-------Main Code for Task-1.1---------'''
# Sequentially go over through 3 images for removing distortion using point-to-point
#Image 1

image_1 = cv2.imread('Img1.JPG')
PQSR_img_1 = np.array([[463,282],[489, 289],[487, 325],[461, 317]])
PQSR_world_1 = np.array([[0,0],[75,0],[75,85],[0,85]])
H1 = find_homography (PQSR_img_1 , PQSR_world_1)
H1_inv = np.linalg.inv(H1)
xmin, ymin, scale, undistImage1 = Bounds_Undistorted(H1,image_1)
undistImage1= transform_image(xmin, ymin, scale, undistImage1, H1_inv,image_1)
cv2.imwrite('Img_1_Results.jpg',undistImage1)
cv2.imshow('Img_1_undistorted',undistImage1)

#Image 2
image_2 = cv2.imread('Img2.jpeg')
PQSR_img_2 = np.array([[473,564],[590, 549],[599, 722],[480, 710]])
PQSR_world_2 = np.array([[0,0],[84,0],[84,74],[0,74]])
H2 = find_homography (PQSR_img_2 , PQSR_world_2)
H2_inv = np.linalg.inv(H2)
xmin, ymin, scale, undistImage2 = Bounds_Undistorted(H2_inv,image_2)
undistImage= transform_image(xmin, ymin, scale, undistImage2,H2_inv,image_2)
cv2.imwrite('Img_2_Results.jpg',undistImage2)
cv2.imshow('Img_2_undistorted',undistImage2)

# Image 3
image_3 = cv2.imread('Img3.JPG')
PQSR_img_3 = np.array([[2055,695],[2670, 715],[2692, 1332],[2088, 1475]])
PQSR_world_3 = np.array([[0,0],[55,0],[55,36],[0,36]])
H3 = find_homography (PQSR_img_3 , PQSR_world_3)
H3_inv = np.linalg.inv(H3)
xmin, ymin, scale, undistImage3 = Bounds_Undistorted(H3,image_3)
undistImage3= transform_image(xmin, ymin, scale, undistImage3, H3_inv,image_3)
cv2.imwrite('Img_3_Results.jpg',undistImage3)
cv2.imshow('Img_3_undistorted',undistImage3)


'''-------------Task 1.2: Vanishing line + Afine-------------------'''
# Image 1
image_1 = cv2.imread('Img1.JPG')
PQSR_img_1 = np.array([[463,282],[489, 289],[487, 325],[461, 317]])    
H_V = find_VL_homography (PQSR_img_1)
H_V_inv = np.linalg.inv(H_V)
xmin, ymin, scale, undistImage_VL = Bounds_Undistorted(H_V,image_1)
undistImage_VL= transform_image(xmin, ymin, scale, undistImage_VL, H_V_inv,image_1)
cv2.imwrite('Img_1_VL.jpg',undistImage_VL)
#Remove affine
H_Aff = ProjectiveCorr_Points(H_V,PQSR_img_1)
H_Aff_inv = np.linalg.inv(H_Aff)
H_VL_Aff = np.dot(H_Aff_inv,H_V)
H_VL_Aff_inv = np.linalg.inv(H_VL_Aff)
xmin, ymin, scale, undistImage_VLAF = Bounds_Undistorted(H_VL_Aff,image_1)
undistImage_VLAF= transform_image(xmin, ymin, scale, undistImage_VLAF, H_VL_Aff_inv,image_1)
cv2.imwrite('Img_1_VL_Aff.jpg',undistImage_VLAF)

# Image 2
image_2 = cv2.imread('Img2.jpeg')
PQSR_img_2 = np.array([[424,231],[514, 167],[514, 298],[425, 346]])
H_V = find_VL_homography (PQSR_img_2)
H_V_inv = np.linalg.inv(H_V)
xmin, ymin, scale, undistImage_VL = Bounds_Undistorted(H_V,image_2)
undistImage_VL= transform_image(xmin, ymin, scale, undistImage_VL, H_V_inv,image_2)
cv2.imwrite('Img_2_VL.jpg',undistImage_VL)
#Remove affine
H_Aff = ProjectiveCorr_Points(H_V,PQSR_img_2)
H_Aff_inv = np.linalg.inv(H_Aff)
H_VL_Aff = np.dot(H_Aff_inv,H_V)
H_VL_Aff_inv = np.linalg.inv(H_VL_Aff)
xmin, ymin, scale, undistImage_VLAF = Bounds_Undistorted(H_VL_Aff,image_2)
undistImage_VLAF= transform_image(xmin, ymin, scale, undistImage_VLAF, H_VL_Aff_inv,image_2)
cv2.imwrite('Img_2_VL_Aff.jpg',undistImage_VLAF)

# Image 3
image_3 = cv2.imread('Img3.jpg')
PQSR_img_3 = np.array([[2058,698],[2658, 715],[2689, 1329],[2094, 1472]])    
H_V = find_VL_homography (PQSR_img_3)
H_V_inv = np.linalg.inv(H_V)
xmin, ymin, scale, undistImage_VL = Bounds_Undistorted(H_V,image_3)
undistImage_VL= transform_image(xmin, ymin, scale, undistImage_VL, H_V_inv,image_3)
cv2.imwrite('Img_3_VL.jpg',undistImage_VL)
# Remove affine
H_Aff = ProjectiveCorr_Points(H_V,PQSR_img_3)
H_Aff_inv = np.linalg.inv(H_Aff)
H_VL_Aff = np.dot(H_Aff_inv,H_V)
H_VL_Aff_inv = np.linalg.inv(H_VL_Aff)
xmin, ymin, scale, undistImage_VLAF = Bounds_Undistorted(H_VL_Aff,image_3)
undistImage_VLAF= transform_image(xmin, ymin, scale, undistImage_VLAF, H_VL_Aff_inv,image_3)
cv2.imwrite('Img_3_VL_Aff.jpg',undistImage_VLAF)


'''------One step method------'''
#Image 1
image_1 = cv2.imread('Img1.jpg')
PQSR_img_1 = np.array([[463,282],[489, 289],[487, 325],[461, 317]])
H_one = find_homography_one(PQSR_img_1)
H_one_inv = np.linalg.inv(H_one)
xmin, ymin, scale, undistImage_one = Bounds_Undistorted(H_one_inv,image_1)
undistImage_one = transform_image(xmin, ymin, scale, undistImage_one, H_one,image_1)
cv2.imwrite('Img_1_One_step.jpg',undistImage_one)

#Image 2
image_2 = cv2.imread('Img2.jpeg')
PQSR_img_2 = np.array([[424,231],[514, 167],[514, 298],[425, 346]])
H_one = find_homography_one(PQSR_img_2)
H_one_inv = np.linalg.inv(H_one)
xmin, ymin, scale, undistImage_one = Bounds_Undistorted(H_one_inv,image_2)
undistImage_one = transform_image(xmin, ymin, scale, undistImage_one, H_one,image_2)
cv2.imwrite('Img_2_One_step.jpg',undistImage_one)

#Image 3
image_3 = cv2.imread('Img3.jpg')
PQSR_img_3 = np.array([[2058,698],[2658, 715],[2689, 1329],[2094, 1472]])
H_one = find_homography_one(PQSR_img_3)
H_one_inv = np.linalg.inv(H_one)
xmin, ymin, scale, undistImage_one = Bounds_Undistorted(H_one_inv,image_3)
undistImage_one = transform_image(xmin, ymin, scale, undistImage_one, H_one,image_3)
cv2.imwrite('Img_3_One_step.jpg',undistImage_one)
"""

'''--------Code For Task 2.1-----------'''
#My image 1
My_image_1 = cv2.imread('My_img_1.jpg')
PQSR_Myimg_1 = np.array([[1168,939],[1721, 941],[1680, 1370],[1049, 1370]])  
#PQSR_world_1 = np.array([[0,0],[2.54,0],[2.54,2.54],[0,2.54]])
PQSR_world_1 = np.array([[0,0],[4.1,0],[4.1,4.1],[0,4.1]])
H3 = find_homography (PQSR_Myimg_1 , PQSR_world_1)
H3_inv = np.linalg.inv(H3)
xmin, ymin, scale, undistImage3 = Bounds_Undistorted(H3,My_image_1)
undistImage3= transform_image(xmin, ymin, scale, undistImage3, H3_inv,My_image_1)
cv2.imwrite('My_Img_1_Results.jpg',undistImage3)
cv2.imshow('My_Img_1_undistorted',undistImage3)

#My image 2
My_image_2 = cv2.imread('My_img_2.jpg')
PQSR_Myimg_2 = np.array([[1009,1393],[1296, 1609],[1321, 2179],[1044, 2006]]) 
PQSR_world_2 = np.array([[0,0],[115,0],[115,140],[0,140]])
H3 = find_homography (PQSR_Myimg_2 , PQSR_world_2)
H3_inv = np.linalg.inv(H3)
xmin, ymin, scale, undistImage3 = Bounds_Undistorted(H3,My_image_2)
undistImage3= transform_image(xmin, ymin, scale, undistImage3, H3_inv,My_image_2)
cv2.imwrite('My_Img_2_Results.jpg',undistImage3)
cv2.imshow('My_Img_2_undistorted',undistImage3)

'''-------------Task 1.2: Vanishing line + Afine-------------------'''
#My image 1
My_image_1 = cv2.imread('My_img_1.jpg')
PQSR_Myimg_1 = np.array([[1168,939],[1721, 941],[1680, 1370],[1049, 1370]])  
H_V = find_VL_homography (PQSR_Myimg_1)
H_V_inv = np.linalg.inv(H_V)
xmin, ymin, scale, undistImage_VL = Bounds_Undistorted(H_V,My_image_1)
undistImage_VL= transform_image(xmin, ymin, scale, undistImage_VL, H_V_inv,My_image_1)
cv2.imwrite('My_Img_1_VL.jpg',undistImage_VL)
# Remove affine
H_Aff = ProjectiveCorr_Points(H_V,PQSR_Myimg_1)
H_Aff_inv = np.linalg.inv(H_Aff)
H_VL_Aff = np.dot(H_Aff_inv,H_V)
H_VL_Aff_inv = np.linalg.inv(H_VL_Aff)
xmin, ymin, scale, undistImage_VLAF = Bounds_Undistorted(H_VL_Aff,My_image_1)
undistImage_VLAF= transform_image(xmin, ymin, scale, undistImage_VLAF, H_VL_Aff_inv,My_image_1)
cv2.imwrite('My_Img_1_VL_Aff.jpg',undistImage_VLAF)

#My image 2
My_image_2 = cv2.imread('My_img_2.jpg')
PQSR_Myimg_2 = np.array([[1009,1393],[1296, 1609],[1321, 2179],[1044, 2006]])
H_V = find_VL_homography (PQSR_Myimg_2)
H_V_inv = np.linalg.inv(H_V)
xmin, ymin, scale, undistImage_VL = Bounds_Undistorted(H_V,My_image_2)
undistImage_VL= transform_image(xmin, ymin, scale, undistImage_VL, H_V_inv,My_image_2)
cv2.imwrite('My_Img_2_VL.jpg',undistImage_VL)
# Remove affine
H_Aff = ProjectiveCorr_Points(H_V,PQSR_Myimg_2)
H_Aff_inv = np.linalg.inv(H_Aff)
H_VL_Aff = np.dot(H_Aff_inv,H_V)
H_VL_Aff_inv = np.linalg.inv(H_VL_Aff)
xmin, ymin, scale, undistImage_VLAF = Bounds_Undistorted(H_VL_Aff,My_image_2)
undistImage_VLAF= transform_image(xmin, ymin, scale, undistImage_VLAF, H_VL_Aff_inv,My_image_2)
cv2.imwrite('My_Img_2_VL_Aff.jpg',undistImage_VLAF)

'''---------One_step Method------------'''
#My image 1
My_image_1 = cv2.imread('My_img_1.jpg')
PQSR_Myimg_1 = np.array([[1168,939],[1721, 941],[1680, 1370],[1049, 1370]])
H_one = find_homography_one(PQSR_Myimg_1)
H_one_inv = np.linalg.inv(H_one)
xmin, ymin, scale, undistImage_one = Bounds_Undistorted(H_one_inv,My_image_1)
undistImage_one = transform_image(xmin, ymin, scale, undistImage_one, H_one,My_image_1)
cv2.imwrite('My_Img_1_One_step.jpg',undistImage_one)

#My image 2
My_image_2 = cv2.imread('My_img_2.jpg')
PQSR_Myimg_2 = np.array([[1009,1393],[1296, 1609],[1321, 2179],[1044, 2006]])
H_one = find_homography_one(PQSR_Myimg_2)
H_one_inv = np.linalg.inv(H_one)
xmin, ymin, scale, undistImage_one = Bounds_Undistorted(H_one_inv,My_image_2)
undistImage_one = transform_image(xmin, ymin, scale, undistImage_one, H_one,My_image_2)
cv2.imwrite('My_Img_2_One_step.jpg',undistImage_one)
