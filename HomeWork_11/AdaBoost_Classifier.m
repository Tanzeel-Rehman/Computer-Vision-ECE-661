function Ada_Classifier = AdaBoost_Classifier(Combined_Features,n_Positive,tpr_thresh,fpr_thresh)
%Function returning the AdaBoost Classifier structure
% Maximum iterations for the Ada-Boosting
T = 20;
%Total and negative samples in current iteration
total_samples = size(Combined_Features,2);
n_Negative = total_samples - n_Positive;

%Compute the weights based on the num of oservations (prior probability)
weight_pos(1:n_Positive,1) = 0.5/n_Positive;
weight_neg(1:n_Negative,1) = 0.5/n_Negative;
%Concatenate weights
Weight_comb = [weight_pos;weight_neg];
%Define the true labels for psoitive and negative classes
label_pos(1:n_Positive,1) = 1;
label_neg(1:n_Negative,1) = 0;
%Concatenate the labels
lbls_comb = [label_pos;label_neg];
%Parameters of AdaBosst Classifier
Clf_params = zeros(T,4);
cascade_results = [];
alphas = [];
TPR = [];
FPR = [];

%Get new features after removing the negative examples with correct
%classification
%Features_new = Combined_Features(:,img_idx);


for i = 1:T
    %Normalize the updated weights at every iteration
    Weight_comb = Weight_comb./sum(Weight_comb);
    %Get the cascade classifier results
    Classifier = Cascade_Classifier(Combined_Features,n_Positive,Weight_comb,lbls_comb);
    %Calculating the parameters for updating the weights
    beta = Classifier.min_Err / (1-Classifier.min_Err);
    a = log(1/beta);
    %Updating the weights for next cascade classifier
    Weight_comb = Weight_comb.*beta.^(1-abs(lbls_comb-Classifier.classification_results));
    %Append the lists for making comaprison
    alphas = [alphas;a];
    cascade_results = [cascade_results,Classifier.classification_results];
    
    %Find the threshold alpha by finding min of class_results*alpha
    C = cascade_results(:,1:i) * alphas(1:i,1);
    threshold_alpha = min(C(1:n_Positive));
    %Find all the observations which are above threshold
    Cx = C >= threshold_alpha;
    
    %Compute the TPR and FNR for current cascade classfier
    tpr = sum(Cx(1:n_Positive))/n_Positive;
    fpr = sum(Cx(n_Positive+1:end))/n_Negative;
    TPR = [TPR;tpr];
    FPR = [FPR;fpr];
    %Terminate the search if current TPR and FPR meets the requirement
    if ((tpr >= tpr_thresh) && (fpr <= fpr_thresh))
        break;
    end
    %Keep the copy of best parameters
    Clf_params(i,:) = [Classifier.Features,Classifier.theta,Classifier.polarity,a];
end

%Now pick only those negative examples which are miscalssified for new
%cascade
[negative_sorted,sorted_idx] = sort(Cx(n_Positive+1:end));
for j = 1:n_Negative
    if negative_sorted(j)>0
    neg_misclassified = sorted_idx(j:end);
    break;
    end
end
%Index containing all positive and miscalssified negative observations
if sum(negative_sorted)>0
    new_idx = [1:n_Positive,neg_misclassified'+n_Positive];
else
    new_idx = 1:n_Positive;
end
new_Features = Combined_Features(:,new_idx);

%Update the classifier structure.
Ada_Classifier.ClassifierParams = Clf_params;
Ada_Classifier.NewIdx = new_idx;
Ada_Classifier.Iterations = i;
Ada_Classifier.FPR = FPR(i);
Ada_Classifier.NewFeatures = new_Features;
end
