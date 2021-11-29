%------This is main function for PCA and LDA face classification ---------%
close all; warning off
%-----Function calls for the PCA classification-----------% 
%number of people in images
n_people=30;
% number of samples per person
n_samples=21;
% number of PCs to be extracted
n_PCs = 20;
%Directories containing the images for train and test datasets
train_imgs = 'ECE661_2020_hw11_DB1/train/';
test_imgs = 'ECE661_2020_hw11_DB1/test/';
%Load and pre-process the training images
[norm_vec_train, Vec_train, m_train] = PreProcess_images(train_imgs, n_people, n_samples);  
%Lod and pre-process the test images
[norm_vec_test, Vec_test, m_test] = PreProcess_images(test_imgs, n_people, n_samples);
%Get the normalized weight vector from the training data
norm_w = PCA_Custom(norm_vec_train);

%Create the class labels 
Class_labels=zeros(n_people*n_samples,1);
for i=1:n_people
    Class_labels((i-1)*n_samples+1:(i-1)*n_samples+n_samples,1)=i;
end

Accuracy_PCA = zeros(1, n_PCs);
for i=1: n_PCs
    latent_PCs=norm_w(:,1:i);
    %Reproject the image vectors from training and test data onto
    %sub-space defined by number of columns of weight matrix (cols grow sequentially)
    training_feat_PCA = latent_PCs' * norm_vec_train;
    test_feat_PCA = latent_PCs' * norm_vec_test;
    %Fit the K-NN classifier on training data with K = 1 and Eucledian distance
    Mdl_PCA=fitcknn(training_feat_PCA', Class_labels, 'distance', 'euclidean','NumNeighbors',1,...
        'NSMethod','exhaustive','BreakTies','smallest');
    %Apply the KNN classifier on test features to get predictions
    pred_test_PCA=Mdl_PCA.predict(test_feat_PCA');
    %Count the total correct predicitions
    Correct_total_PCA = sum(pred_test_PCA==Class_labels);
    Accuracy_PCA(1,i) =(Correct_total_PCA/(n_people*n_samples))*100;
end

% %Plot the accuracy against the n_PCs
% plot((1:n_PCs),Accuracy_PCA);

%-------Function Calls for the LDA---------------%
Accuracy_LDA = zeros(1, n_PCs);

[uw, Z] = LDA_custom(Vec_train,m_train,n_people,n_samples);
w = Z * uw;
normw = w./vecnorm(w);
for i = 1:n_PCs
    latent_LDs = normw(:,1:i);
    %Reproject the image vectors from training and test data onto
    %sub-space defined by number of columns of weight matrix (cols grow sequentially)
    training_features_LDA = latent_LDs' * (Vec_train - m_train);
    test_features_LDA = latent_LDs' * (Vec_test -m_test);
    %Fit the K-NN classifier on training data with K = 1 and Eucledian distance
    MDL_LDA=fitcknn(training_features_LDA', Class_labels, 'distance', 'euclidean','NumNeighbors',1,...
        'NSMethod','exhaustive','BreakTies','smallest');

    %Apply the KNN classifier on test features to get predictions
    pred_test_LDA=MDL_LDA.predict(test_features_LDA');
    %Count the total correct predicitions
    Correct_total_LDA = sum(pred_test_LDA==Class_labels);
    Accuracy_LDA(1,i) =(Correct_total_LDA/(n_people*n_samples))*100;  
end 
%Plot the accuracy against the n_PCs
plot((1:n_PCs),Accuracy_PCA,'g-*','DisplayName','PCA');
hold on;
plot((1:n_PCs),Accuracy_LDA,'b--o','DisplayName','LDA');
legend;
hold off;
xlabel('Number of Latentspace dimensions');
ylabel('Accuracy (%)');