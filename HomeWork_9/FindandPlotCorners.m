function Imagecorners_sorted = FindandPlotCorners(filename,h_res,can_thresh)    
% Read the color image,convert to gray scale and perform canny edge
% detection

ColorImg = imread(filename);
gray_img = rgb2gray(ColorImg);
edges = edge(gray_img,'canny',can_thresh); 
% Show the images with edges    
figure()
imshow(edges)
    
%--- Start to perform hough transformation----% 
[H, theta, rho] = hough(edges,'RhoResolution',h_res);

% There are 5 pairs of horizontal and 4 pairs of verticle lines, so total
% 18 lines. These are 18 strongest lines.
peaks = houghpeaks(H,18,'threshold',ceil(0.2*max(H(:)))); 
% Pick the bigger fillgap value otherwise many small lines will appear.
% Discard some small lines
lines = houghlines(edges,theta,rho,peaks,'FillGap',150,'MinLength',70);

%Plot the lines
figure()
imshow(ColorImg)
hold on
for k = 1:length(lines)
    pt1 = lines(k).point1;
    pt2 = lines(k).point2;
    % Find the vertical lines and plot them along the entire image
    if abs(lines(k).theta) <=1 %Allowence for lines whose theta is not exactly zero
        y = 1:size(gray_img,1);
        x = pt1(1,1)*ones(1,length(y));
        plot(x,y,'Color','blue');
    % Find the horizontal lines and plot them along the entire image. 
    %Can't fix their position as boxes are inclined in x direction for multiple images. 
    else
        % Find the line parameters   
        m = (pt1(1,2)-pt2(1,2))/(pt1(1,1)-pt2(1,1));
        c = pt1(1,2) - m*pt1(1,1);
        x = 1:size(gray_img,2);
        y =  m*x + c;
        plot(x,y,'Color','blue');
    end
end

%Find the corners and plot them
Imagecorners_sorted = FindCorners(lines,gray_img);

for i = 1:length(Imagecorners_sorted)
    hold on
    text(Imagecorners_sorted(i,1),Imagecorners_sorted(i,2),int2str(i),'Color','r');
end
hold off;
end
