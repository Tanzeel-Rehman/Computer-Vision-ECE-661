%----This is main function for training and testing AdaBoost Classifier--%

%First of all compute Features
%Training features
% fprintf('Now Extracting features\n');
% Features_data = ComputeDatasetFeatures('ECE661_2020_hw11_DB2\train\positive\','Train_Positive.mat');
% Features_data = ComputeDatasetFeatures('ECE661_2020_hw11_DB2\train\negative\','Train_Negative.mat');
% %Test Features
% Features_data = ComputeDatasetFeatures('ECE661_2020_hw11_DB2\test\positive\','Test_Positive.mat');
% Features_data = ComputeDatasetFeatures('ECE661_2020_hw11_DB2\test\negative\','Test_Negative.mat');
%Train the network 
Num_FP = train_AdaBoost('Train_Positive.mat','Train_Negative.mat','Trained_Classifier.mat');
%Get the FPR and FNR from test dataset
[FPR_test,FNR_test] = test_AdaBoost('Test_Positive.mat','Test_Negative.mat','Trained_Classifier.mat');