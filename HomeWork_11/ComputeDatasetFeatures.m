function Features_data = ComputeDatasetFeatures(filepath,savename)
%Struct contining all the png files in a directory
img_set = dir([filepath,'*.png']);

n_imgs = length(img_set); 
Features_data = zeros(8000+3900,n_imgs);
for i = 1:n_imgs
    Color_img = imread([filepath, img_set(i).name]);
    Gray_img = double(rgb2gray(Color_img));
    Haar_img = integralImage(Gray_img);
    Features_data(:,i) = ComputeFeatures(Haar_img);
end
%Dataset_Features.features = Features_data;

%Save the Computed features of dataset
save(savename,'Features_data','-v7.3');
end