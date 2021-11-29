function All_Features = ComputeFeatures(Haar_img)
%%Function responsible for computing the features from an input Haar image
%Get the size of Haar image. This will be +1 pixel compared to original
%image size
[img_height,img_width] = size(Haar_img);
Horizontal_Features = ComputeHorizontalFeatures(Haar_img,img_width,img_height);
Vertical_Features = ComputeVerticalFeatures(Haar_img,img_width,img_height);
All_Features = [Horizontal_Features;Vertical_Features];
end

function Features=ComputeHorizontalFeatures(Haar_img,img_width,img_height)
Features = [];
kernel_sizes = 2:2:img_width-1;
%Control the number of kernels
for n  = 1:length(kernel_sizes)
    kernel = kernel_sizes(n);
    for i = 1:img_height-1 %Height
        for j = 1:(img_width-1 - kernel + 1)%Width
            kh = kernel/2;  %Half_kernel 
            %Compute the two sums
            Sum_1 = ComputeSum([i;j; i;(j+kh);(i + 1); j;(i + 1);(j+kh)],Haar_img);
            Sum_2 = ComputeSum([i;(j+kh);i;(j+kernel);i+1;(j+kh);i+1;j+kernel],Haar_img);
            Features = [Features; (Sum_2 - Sum_1)];
        end
    end
end
end
function Features = ComputeVerticalFeatures(Haar_img,img_width,img_height)
Features =[];
%Control the number of kernels
kernel_size = 2:2:img_height-1;
for n  = 1:length(kernel_size)
    kernel = kernel_size(n);
    for i = 1:(img_height -1 - kernel + 1)%Height
        for j = 1:img_width - 2 %Width
            kh = kernel/2;  %Half_kernel 
            %Compute the two sums
            Sum_1 = ComputeSum([i;j;i;j+2;i+kh;j;i+kh;j+2],Haar_img);
            Sum_2 = ComputeSum([i+kh;j;i+kh;j+2;i+kernel;j;i+kernel;j+2],Haar_img);
            Features = [Features; (Sum_1 - Sum_2)];
        end
    end
end
end
function Pixel = ComputeSum(Corners,Harr_Img)
    cor1 = Harr_Img(Corners(1), Corners(2));
    cor2 = Harr_Img(Corners(3), Corners(4));
    cor3 = Harr_Img(Corners(5), Corners(6));
    cor4 = Harr_Img(Corners(7), Corners(8));
    Pixel = cor4 + cor1 - cor2 - cor3;
end 