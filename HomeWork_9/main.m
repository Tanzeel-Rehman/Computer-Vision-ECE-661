%----------- This is main fuunction--------%
close all; clc; clear all;
%Flag to indicate whether distortion needs to be included in the model or
%not.
distort_flag = 1;
box_size = 25; % blocksize mm 

%Parameters of dataset1
n_images = 40; 
direc = '..\Files\Dataset1\';
h_res = 0.6; can_thresh = 0.8;
standard_img = 11; 
backproj_imgs = [13, 15, 21, 28]; 

%Parameters of dataset2
% n_images = 20;
% direc = '..\Files\Dataset2';
% h_res = 0.7; can_thresh = 0.7;
% standard_img = 1; 
% backproj_imgs = [4, 8, 13, 15]; 

% Get the world coordinates of the checkerboard pattern using the box
% dimensions physically measured in mm
coords_world = find_world_coords(box_size);
%Compute the corners, homographies and V matrices for all images
[H_Dataset,Corners_Dataset,V_Dataset] = ComputeDatasetProperties(n_images,direc,h_res,can_thresh,coords_world);
% Find the intrinsic parameters of the camera using linear least squares
[U,D,V] = svd(V_Dataset);
b = V(:,6); 
K = Find_K(b);

% Find the Extrinsic parameters of images using linear least squares
R_LLs = cell(1,n_images);
t_LLs = cell(1,n_images);
intial_estimates_vec = zeros(1,n_images*6);
for k = 1:n_images
    H = H_Dataset{k};
    [R,t] = Find_Extrinsic(H,K);
    R_LLs{k} = R;
    t_LLs{k} = t;
	w = R_Rodriguez(R);
    intial_estimates_vec(1,(k-1)*6+1:k*6) = [w',t'];
end
%Add the initial values of k1 and k2 if distortion needs to be corrected
if (~distort_flag)
    intial_estimates_vec =[intial_estimates_vec,K(1,1:3),K(2,2:3)];
else
    intial_estimates_vec =[intial_estimates_vec,K(1,1:3),K(2,2:3),0,0];
end

%Optimize the intrisic and extrinsic parameters using Levnberg algorithm
opts = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt');
final_estimates_vec = lsqnonlin(@func_LM_cost,intial_estimates_vec,[],[],opts,coords_world,Corners_Dataset,distort_flag,n_images);

%Optimized intrisic parameters
K1 = [final_estimates_vec(1,6*n_images+1:6*n_images+3); 0,final_estimates_vec(1,6*n_images+4:6*n_images+5); 0, 0, 1];
if(distort_flag)
    k1 = final_estimates_vec(1,6*n_images+6);
	k2 = final_estimates_vec(1,6*n_images+7);
end

%Get Extrinsic Parameters using optimized estimates
R_LM = cell(1,n_images);
t_LM = cell(1,n_images);
for k = 1:n_images
     wt = final_estimates_vec(1,(k-1)*6+1:k*6);
     omega = wt(1,1:3);
     t_LM{k} = wt(1,4:6)';
     R_LM{k} = Rodriguez_R(omega);      
end
%Initialize the stats for checking the backprojection quality
residual_LL = zeros (length(backproj_imgs),2);
residual_LM = zeros (length(backproj_imgs),2);
for i = 1:length(backproj_imgs)
    residual_LL(i,:) = ReprojectCorners(R_LLs,t_LLs,K,Corners_Dataset,direc,standard_img,backproj_imgs(i));
    residual_LM(i,:) = ReprojectCorners(R_LM,t_LM,K1,Corners_Dataset,direc,standard_img,backproj_imgs(i));
end