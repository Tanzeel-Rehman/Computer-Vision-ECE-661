function [norm_vec_all,Vec_imgs_all,m_vec] = PreProcess_images(filePath,n_people,n_samples)
%%Function responsible for pre-process the images to vectorize them and compute mean.
%Normalize the vector of images.
%%
%Image dimensions
height = 128; width = 128;
%Size of dataset:
dataset_size = n_people*n_samples;
%Output vector for the entire dataset
Vec_imgs_all = zeros(height*width,dataset_size);
%Read all images one by one, convert them to grayscale and reshape them
%to 1D vector
for i = 1:n_people
 for j = 1:n_samples
     Color_image = imread([filePath,num2file(i),'_',num2file(j),'.png']);
     gray_img = rgb2gray(Color_image);
     Vec_img = double(reshape(gray_img,height*width,1));
     %Normalize the image vector
     Vec_img = Vec_img/norm(Vec_img);
     %Accumulate the vectors from all the images
     Vec_imgs_all(:,(i-1)*n_samples+j) = Vec_img;
 end
end
%Compute the mean of a vector representing all image in dataset
m_vec = mean(Vec_imgs_all,2);
%Subtract mean
norm_vec_all = Vec_imgs_all - m_vec;
end

function filename = num2file(n)
%If the number passed is less than 10, add precceeding 0
if n <10
    filename = num2str(n,'%02.f');
else
    %If passed is >10, procced as it is
    filename = num2str(n);
end
end