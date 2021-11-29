function error = func_LM_cost(intial_estimates_vec,worldCoord,pixelCoord,dist_falg,num_imgs)
%Get initial values of intrinsic parameters
K = [intial_estimates_vec(6*num_imgs+1:6*num_imgs+3); 0, intial_estimates_vec(6*num_imgs+4:6*num_imgs+5); 0, 0, 1]; 
% Find k1 and k2 if the radial distortion is needed
if(dist_falg == 1)
    k1 = intial_estimates_vec(6*num_imgs+6);
	k2 = intial_estimates_vec(6*num_imgs+7); 
	K1 = [intial_estimates_vec(6*num_imgs+1), 0, intial_estimates_vec(6*num_imgs+3); 0, intial_estimates_vec(6*num_imgs+4:6*num_imgs+5); 0 0 1];
end

error = zeros(num_imgs*160,1);
%loop through all the images
for k = 1:num_imgs
     wt = intial_estimates_vec(1,(k-1)*6+1:k*6);
     omega = wt(1,1:3);
     t = wt(1,4:6)';
     R = Rodriguez_R(omega);     
     projPixelCoord = zeros(80,2);
     %loop throough all the points in an images
     for i = 1:80
         x = K * [R t] *[worldCoord(i,:), 0, 1]';
         projPixelCoord(i,:) = [x(1)/x(3), x(2)/x(3)];
         %Correct the radial distortion of all the points if it is checked
         if(dist_falg == 1)
            xp = [projPixelCoord(i,:), 1];
            x_hat = pinv(K1)*xp';
            r2 = x_hat(1)^2 + x_hat(2)^2;
            x_rad = x_hat(1) + x_hat(1)*(k1*r2+k2*r2^2);
            y_rad = x_hat(2) + x_hat(2)*(k1*r2+k2*r2^2);
            x = K1*[x_rad y_rad 1]';
            projPixelCoord(i,:) = [x(1)/x(3) x(2)/x(3)];
         end
     end
     e = pixelCoord{k}-projPixelCoord;
     error((k-1)*160+1:k*160) = [e(:,1);e(:,2)];
end
end