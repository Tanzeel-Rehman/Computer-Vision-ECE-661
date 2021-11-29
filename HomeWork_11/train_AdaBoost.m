function Num_FP=train_AdaBoost(Positive_Feat_file,negative_Feat_file,savename)
% Load the stored features for positive and negative examples
Features_positive_train = load(Positive_Feat_file);
Features_negative_train=load(negative_Feat_file);
Feat_positive = Features_positive_train.Features_data;
Feat_neagative = Features_negative_train.Features_data;
%Num of positive and negative samples in the data
Positive_samples = size(Feat_positive,2);
Negative_samples = size(Feat_neagative,2);

%Concatenate all the features
Combined_Features = [Feat_positive,Feat_neagative];
%Build Bossted classifiers
for i = 1:10
    Clss_stages(i,1)= AdaBoost_Classifier(Combined_Features,Positive_samples,1,0.5);  
    % Get the new updated features for next iterations
    Combined_Features = Clss_stages(i,1).NewFeatures;
    %Save the fsalse psotitive observations for plotting
    Num_FP(i,1) = size(Combined_Features,2) - Positive_samples; 
    fprintf('Now Running at Stage %s\n', num2str(i));
    fprintf(['Fasle Positives = ', num2str(Num_FP(i,1))]);
    fprintf('\n');
    %Stop if only positive features are left in Combined features
    if (size(Combined_Features,2) == Positive_samples)
        break;
    end
end
plot (1:length(Num_FP),Num_FP/Negative_samples,'g-');
xlabel('Cascade Stages');
ylabel('FPR (%)');
save(savename,'Clss_stages','-v7.3');
end