function [H_Dataset,Corners_Dataset,V_Dataset] = ComputeDatasetProperties(n_images,direc,h_res,can_thresh,world_coord)
%Store the corners,homographies and V matrices for the entire dataset
Corners_Dataset = cell(1,n_images);
H_Dataset = cell(1,n_images);
V_Dataset = zeros(n_images*2,6);

for i = 1:n_images
    filename = strcat([direc, '\Pic_' num2str(i), '.jpg']);
    %Find and plot the coners of images 
    Corners_Dataset{i} = FindandPlotCorners(filename,h_res,can_thresh);
    % Check if the corners are not 80 and report as warning
    if length(Corners_Dataset{i}) ~= 80
        fprintf('Number of corners not detected correctly for image number %d \n',i);
    end
    % Find the homography between the image and world coordinates 
    H_Dataset{i} = find_homography(world_coord,Corners_Dataset{i});
    %Find the V (2*6) matrix for every images
    V = Find_V_Matrix(H_Dataset{i});
    V_Dataset((i-1)*2+1:i*2,:) = [V(2,:);(V(1,:)-V(3,:))];
end
end