function [R,t] = Find_Extrinsic(H,K)
% Get the columns of the homography matrix
h1 = H(:,1); 
h2 = H(:,2); 
h3 = H(:,3);
%Get the K inverse
K_inv = pinv(K);
% Compute the scaling factor from the orthonormality condition of R
scale = 1/norm(K_inv*h1);
% Get the unscaled t vector
t = K_inv * h3;
% If 3 element of t is negative scale should be negative
if t(3) < 0
    scale = -scale;
end
%Find the scaled elements of rotation matrix for the image under consideration
r1 = scale*K_inv*h1;
r2 = scale*K_inv*h2;
r3 = cross(r1,r2);
R = [r1, r2, r3];
%Condition the R matrix
R = Condition_R(R);
% Scale the t vector
t = scale * t;

end

function R=Condition_R(R)
    %Decompose R and reset all singular values to 1
    [U,D,V] = svd(R);
     R = U * V';
end