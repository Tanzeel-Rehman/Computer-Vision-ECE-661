function V = Find_V_Matrix(H)
%Function for finding the Vij from Homography matrix. 
%Inputs:
%   H: Homography matrix between world and pixel coordinates
%Outputs: 
%   V: A matrix containing the Vij. V11,V12 and V22 are at 1st,2nd and 3rd
%   row, respectively.
ij = [1,1;1,2;2,2];
V = zeros(3,6);
for k=1:length(ij)
    i = ij(k,1);
    j = ij(k,2);
    V(k,:) = [H(1,i)*H(1,j), H(1,i)*H(2,j)+H(2,i)*H(1,j),...
                H(2,i)*H(2,j), H(3,i)*H(1,j)+H(1,i)*H(3,j),...
                H(3,i)*H(2,j)+H(2,i)*H(3,j),H(3,i)*H(3,j)];
end
end