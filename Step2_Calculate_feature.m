%% Step 2: This script is to calculate the features from the PPG data for all particiapants
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
%                  Feature Engineering: PPG Point Process                %
%                                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Make sure to run Final_parameter_selection.m first
% This scripts models the RR intervals as point process, extract time and
% spectral features and creates the dataset (Features + Labels) for trainig the machine
% learning models
% Point process algorithms scripts are derived from http://users.neurostat.mit.edu/barbieri/pphrv
%
% Author: Akshay Sujatha Ravindran
% email: akshay dot s dot ravindran at gmail dot com



load('parameter_selection.mat')
Fs          = 64;  % Sampling rate 
delta       = 1/Fs; % time increment in updating parameters (in seconds)
file_names  = dir('.\data_preprocessed_matlab') % Replace with the directory containing the files
ppg_column  = 39; % Column number corresponding to PPG data
pcm_each    = P(45:66,:); % Identified model orders
for i = 45:66
    sub = i-44;
    load(file_names(sub+2).name)
    s = str2num(file_names(sub+2).name(2:3))
    
    % Iterate through all the trials
    for trial = 1:40   
        % load the ppg
        ppg     = squeeze(data(trial, ppg_column, :));
        
        % Downsample to 64 Hz sampling rate        
        ppg     = downsample(ppg,2);
        
        
        % High pass filter the PPG signal (4th order 0.5 Hz)
        [a,b]   = butter(4,0.5/(Fs/2),'high');        
        ppg     = filtfilt(a,b,double(ppg));
        
        % Low pass filter the PPG signal (4th order 5 Hz)
        [a,b] = butter(4,5/(Fs/2),'low');  
        ppg   = filtfilt(a,b,double(ppg));

        % Find the peaks in the PPG signal with additional condition to
        % minimize false positives
        [~, peak_times]      = findpeaks(-ppg,'MinpeakDistance', Fs / 2, 'MinPeakProminence', 100,'MinPeakHeight',100);     
        peak_times           = peak_times./Fs;             
        p                    = pcm_each(s,trial); % Previously identified best model order   
        
        % Compute the history dependent inverse gaussian regression
                % parameters by maximizing the likelihood
        [Thetap,Kappa,opt_old]= regr_likel(peak_times,p);        
        
        % Compute the time varying parameters at each moment in time with
        % delta resolution
        [Mu,opt]              = pplikel_ASR(peak_times,[opt_old.Theta0;Thetap],delta,p);
        
        
        Var                   = opt.meanRR.^3 / Kappa; 
        Var                   = 1e6 * Var; % from [s^2] to [ms^2]  
        thetap_rep            = repmat(Thetap,1,length(Mu));  
        loc_nan               = find(isnan(Mu));
        thetap_rep(:,loc_nan) = nan;        
        
        % Compute spectral features
        [powLF, powHF, bal, warn, powVLF, powTot] = hrv_indices(thetap_rep, Var, 1./opt.meanRR);
        
        % Different emotional ratings as labels
        Labels(s,trial,1)      = labels(trial,1);
        Labels(s,trial,2)      = labels(trial,3);
        Labels(s,trial,3)      = labels(trial,2);
        Labels(s,trial,4)      = labels(trial,4);  
        

        loc_notnan             = find(~isnan(Mu));
        Features_HR{s,trial}   = [powLF(loc_notnan)', powHF(loc_notnan)', bal(loc_notnan)',...
            powVLF(loc_notnan)', powTot(loc_notnan)',Var(loc_notnan)',Mu(loc_notnan)'];
      
    end
end
save('Features_final.mat','Features_HR','Labels')

