function BestClassifier = Cascade_Classifier(Combined_Features,n_Positive,Weights,lbls)
%% Function returns a classifier structure.
%Total samples obtained from combined psotive and negative samples 
total_samples = size(Combined_Features,2);
%Sum of weights for positive and negative examples
Total_positive = sum(Weights(1:n_Positive,1));
Total_negative =sum(Weights(n_Positive+1:end));
BestClassifier.min_Err = inf;

for i = 1:length(Combined_Features)
    %Initialize the clasisification results
    Class_results = zeros(total_samples,1);
    %Get idx of sorting all the samples with respect to a feature.
    [sorted_feats,idx] = sort(Combined_Features(i,:));
    sorted_weight = Weights(idx);
    sorted_lbl = lbls(idx);
    %Compute the cummulative sume of the sorted weights
    Sum_positive = cumsum(sorted_weight.*sorted_lbl);
    Sum_negative = cumsum(sorted_weight) - Sum_positive;
    %Compute two types of error
    Err_1 = Sum_positive + (Total_negative - Sum_negative);
    Err_2 = Sum_negative + (Total_positive - Sum_positive); 
    %Find the minimum of two errorz
    minErr = min(Err_1,Err_2);
    %The minimum value and index of minimum of two erros
    [min_min_Err, Err_idx] = min(minErr);
    %Calssify the samples.
    if Err_1(Err_idx) > Err_2(Err_idx) 
       %All values are correctly classified upto error index, rest are
       %falsely classified, leaving them zero.
       Polarity = 1;
       Class_results(1:Err_idx,1) = 1; 
       Class_results(idx) = Class_results;
    else
       %All values are correctly classified after error index, before that
       %index all values are falsely classfiied, leaving them zeo.
       Polarity = -1;
       Class_results(Err_idx + 1:end,1) = 1; 
       Class_results(idx) = Class_results;
    end
    
    if Err_idx == 1
         BestClassifier.theta = sorted_feats(1) - 0.5;
    elseif Err_idx == size(Combined_Features,1)
         BestClassifier.theta = sorted_feats(size(Combined_Features,1)) + 0.5;
    else
         BestClassifier.theta = mean([sorted_feats(Err_idx), sorted_feats(Err_idx-1)]);
    end
     %Check if the current error is lower than earlier error to find new best.
     if min_min_Err < BestClassifier.min_Err
         %Update the classifier structure. 
         BestClassifier.min_Err = min_min_Err;
         BestClassifier.classification_results = Class_results;
         BestClassifier.Features = i;
         BestClassifier.polarity = Polarity;

     end
end
end