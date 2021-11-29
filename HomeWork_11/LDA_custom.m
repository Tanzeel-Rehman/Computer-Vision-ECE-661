function [dW,Z] = LDA_custom( Vec_train, m_train, n_people, n_samples )
%Image dimensions
height = 128; width = 128;
%Get Class_wise mean
mean_class = zeros(height*width,n_people);
V_mean = zeros(height*width,n_people*n_samples);
for i = 1:n_people
  mean_class(:,i)=mean(Vec_train(:,(i-1)*n_samples+1:(i-1)*n_samples+n_samples),2);
  V_mean(:,(i-1)*n_samples+1:(i-1)*n_samples+n_samples)= Vec_train(:,(i-1)*n_samples+1:(i-1)*n_samples+n_samples) - mean_class(:,i);
end
%Compute the difference between class and entire training set mean 
Diff_of_Means = mean_class - m_train;
%Decompose between class scatter matrix
[dB,uB] = eig(Diff_of_Means' * Diff_of_Means);
[~,idx] = sort(-1 .* diag(uB));
%dB = dB(:,idx);
%uB = uB(idx);

%Compute the vector V
V = Diff_of_Means * dB;
DB = diag(diag(uB.^(-0.5)));
Z = V*DB;

%Within_Class_scatter
SW = Z' * V_mean;
[dW,uW] = eig(SW*SW');
[~,idx] = sort(diag(uW));
dW = dW(:,idx);
end
