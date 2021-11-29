function A = A_Matrix_Generator(domain_point,range_point)
%function for generating a 2 x 9 design matrix needed to compute Homography
%Inputs:
%   domain_point: Coordinates of a point in the domain image (x,y)
%   range_point: Coordinates of corresponding point in the range image (x',y')
%Output: 
%   A: A 2 x 9 design matrix 
x =domain_point(1,1); y=domain_point(1,2);
xr =range_point(1,1); yr=range_point(1,2);

A = [0,0,0,-x,-y,-1,yr*x,yr*y,yr ; x,y,1,0,0,0,-xr*x,-xr*y,-xr];

end