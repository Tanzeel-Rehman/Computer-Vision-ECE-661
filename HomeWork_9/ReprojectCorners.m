function residuals = ReprojectCorners(R,t,K,Corners_Dataset,dirLoc,standard_img,proj_img)

%Parameters of the image that needs to be projected
P = K * [R{proj_img}(:,1:2) t{proj_img}];
corners_proj_img = Corners_Dataset{proj_img};
corners_proj_img = [corners_proj_img ones(size(corners_proj_img,1),1)];

% Read the image file which needs to be back projected
filename = strcat([dirLoc, '/Pic_' num2str(standard_img), '.jpg']);
Colorimg = imread(filename);
img = rgb2gray(Colorimg);
%Parameters of the fixed image
P_stand = K * [R{standard_img}(:,1:2) t{standard_img}];
% corners of fixed image
corners_fixed = Corners_Dataset{standard_img};
%Projected corners
worldCoordImg = pinv(P) * corners_proj_img';
Corners_Projected = (P_stand * worldCoordImg)';

figure
imshow(Colorimg)
%Loop trhough all the corners
for i = 1:80
    %HC to XY
	Corners_Projected(i,:) = Corners_Projected(i,:) / Corners_Projected(i,3);
	hold on
	%plot the circle of size 16 at corners in green and red fro projected
    plot(uint64(corners_fixed(i,1)),uint64(corners_fixed(i,2)),'g.','MarkerSize',16);
	plot(uint64(Corners_Projected(i,1)),uint64(Corners_Projected(i,2)),'r.','MarkerSize',16);
end
hold off
%Compute vital statistics to evalute the performance of computed parmeters
diff = corners_fixed-Corners_Projected(:,1:2);
%Calculating moments of the error
mean_err = mean(abs(diff(:))); 
var_err = var(abs(diff(:)));
residuals = [mean_err, var_err];
end