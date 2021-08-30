%% Step 1: This script is to estimate the optimal P value using BIC condition
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
%                  Model order selection using BIC condn                 %
%                                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Determine the best model order by using AIC/BIC condition
%  Point process algorithms scripts are derived from http://users.neurostat.mit.edu/barbieri/pphrv
%
% Author: Akshay Sujatha Ravindran
% email: akshay dot s dot ravindran at gmail dot com


addpath('.\dependencies')
% initialization
max_limit   = 10; % Max p value to iterate on
Fs          = 64; % Sampling rate 
addpath('.\data_preprocessed_matlab')
file_names  = dir('.\data_preprocessed_matlab') % Replace with the directory containing the files
ppg_column  = 39; % Column number corresponding to PPG data
%  Iterate over the first 22 participants (First 2 files are not signal files)


for sub = 3:24   
    load(file_names(sub).name) 
    fprintf('Processing sub = %i...\n', sub);    
    % Loop over all the trials  
    for trial =1:40     
        % load the ppg
        ppg = squeeze(data(trial, ppg_column, :)); 
        
        ppg=downsample(ppg,2);
        
        % High pass filter the PPG signal (4th order 0.5 Hz)
        [a,b]=butter(4,0.5/(Fs/2),'high');  
        ppg=filtfilt(a,b,double(ppg));
        
        % Low pass filter the PPG signal (4th order 5 Hz)
        [a,b]=butter(4,5/(Fs/2),'low');  
        ppg=filtfilt(a,b,double(ppg));
        
        
        
        
        % Find the peaks in the PPG signal with additional condition to
        % minimize false positives
        [~, peak_times]=findpeaks(-ppg,'MinpeakDistance', Fs / 2, 'MinPeakProminence', 100,'MinPeakHeight',100); 
        peak_times = peak_times./Fs; % Convert to times in seconds
        
        % Iterate through different p values to find the optimal p
        for p = 2:max_limit
        
            clearvars -except ppg_column peak_times sub AIC_ml_val AICc_ml_val  BIC_ml_val...
                Fs p Thetap Data  pbm_each pm_each pcm_each max_limit ppg_indx data trial file_names   
                
                
                % Compute the history dependent inverse gaussian regression
                % parameters by maximizing the likelihood
                [Thetap,Kappa,opt_old] = regr_likel(peak_times,p);
                
                
                K=p+1; % Total number of parameters
                n = length(peak_times); % Total number of PPG peaks   
                % Compute AIC, AIC_corrected and BIC
                AICc_ml_val(p-1,trial) = -2*(nanmean(opt_old.loglikel)) + (2*K*(K+1))/(n-K-1);
                AIC_ml_val(p-1,trial)=-2*(nanmean(opt_old.loglikel))+2*(K);
                BIC_ml_val(p-1,trial)= -2*nanmean(opt_old.loglikel) + K*log(length(peak_times));               
        end
    end    
    % Find the model with the lowest AIC/BIC values   
    [~, idxm] = min(AIC_ml_val);
    [~, idxcm] = min(AICc_ml_val);
    [~, idxbm] = min(BIC_ml_val);    
    s=str2num(file_names(sub).name(2:3));
    pm_each(s,:) = (idxm) + 1 ;
    pcm_each(s,:) = (idxcm) + 1 ;
    pbm_each(s,:) = (idxbm) + 1 ;     
    clear   AIC_ml_val AICc_ml_val  BIC_ml_val 
end
P=[ pm_each; pcm_each; pbm_each;];
save('parameter_selection.mat','P')


