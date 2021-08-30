%% Step 3: Prepare the dataset for training the ML models
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
%                  Prepare dataset for ML model training                 %
%                                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script zscores the features and then create subject specific dataset
% with overlapping windows for training the machine learning models
%
% Make sure to run Final_parameter_selection.m and Estimate_features.m first
%
% Author: Akshay Sujatha Ravindran
% email: akshay dot s dot ravindran at gmail dot com

%% Zscore the features per subject
% Different features have different value ranges, standardize to make the model
% unbiased to any specific features
load('Features_final.mat')
for sub = 1:22 % Loop through all subjects    
    Concatenated_Features=[];
    % Concatenate all trials to zscore w.r.t. entire dataset
    for iter = 1:40 
         loc_start(iter)       = length(Concatenated_Features)+1; % start pos for trial 
         tempHR                = Features_HR{sub,iter}; 
         Concatenated_Features = [Concatenated_Features;tempHR]; % Concatenate Features
         loc_end(iter)         = length(Concatenated_Features);  % End pos for trial 
    end
    % zscore the features
    zscored_features           = zscore(Concatenated_Features);   
    for iter = 1:40        
        Features_HR{sub,iter}  = zscored_features(loc_start(iter):loc_end(iter),:);  
    end 
end

%% Window the continuous features and create the dataset
non_overlap = 3;
seg_len     = 320;
k           = 1;
for class=1:3    % Loop through each conditions     
    for sub=1:22     % Loop through each subjects     
        % Initialize the variables
        Feature         = zeros(60000,seg_len,6); 
        labels          = zeros(60000,1);
        identifier_val  = zeros(60000,2);
        k               = 1;
        Condition_label                             = squeeze(Labels(sub,:,class));
        
        % Binarize the ratings to serve as output labels
        Condition_label(find(Condition_label<5))    = 0;
        Condition_label(find(Condition_label>=5))  = 1;  
        
        for iter=1:length(Condition_label)
            tempHR=Features_HR{sub,iter};%    Extract per trial/subject continuous feature set         
            % Segment the continuous feature set into overlapped windows of length
            % seg_len with non_overlap number of samples not being overlapped
            for num = 0:floor((length(tempHR)/non_overlap)-(seg_len/non_overlap))
                % Do not include 'bal' feature            
                seg_HR              = tempHR((num*non_overlap)+1:(num*non_overlap+seg_len),[1,2,4,5,6,7]); 
                Feature(k,:,:)      = seg_HR;  
                labels(k)           = Condition_label(iter);
                identifier_val(k,1) = iter;
                identifier_val(k,2) = sub; % Subject identifier                
                k                   = k+1;
            end
        end        
        % Remove the empty rows
        Feature=Feature(1:k-1,:,:);
        labels=labels(1:k-1);
        identifier_val=identifier_val(1:k-1,:);   
        
        % Save the dataset        
        save(['Data_Class_',num2str(class),'_Subject_',num2str(sub)],'Feature','labels','identifier_val')
        clear Feature identifier_val labels
    end 
end
 

            