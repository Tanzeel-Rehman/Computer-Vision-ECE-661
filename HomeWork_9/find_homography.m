function H = find_homography(Domain_pts,Range_pts)
%function for estimating the 3 x 3 Homography matrix---Modified from HW2
%Inputs:
%   Domain_pts: An n x 2 array containing coordinates of domian image points(Xi,Yi)
%   range_point: An n x 2 array containing coordinates of range image points(Xi',Yi')
%Output: A 3 x 3 Homography matrix 

%Find num of points provided
n = length(Domain_pts);
%Initialize A Design matrix having size of 2n x 8
A = zeros(2*n,9);
%Loop through all the points provided and stack them vertically, this will result in 2n x 9 Design matrix
for i=1:n
    A((i-1)*2+1:i*2,:)=A_Matrix_Generator(Domain_pts(i,:),Range_pts(i,:));
end
%Decompose A using SVD decomposition
[U,D,V] = svd(A);
h = V(:,9); %Eigen vector corresponding to the smallest eigen value of D

% Rearrange the vector h to Homography matrix H
H = [h(1:3)'; h(4:6)'; h(7:9)'];
end