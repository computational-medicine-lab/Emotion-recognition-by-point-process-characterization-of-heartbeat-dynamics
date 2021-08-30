%% Train an ensemble model to serve as baseline model
for class=1:3
    for sub=1:22
        % Load the subject and class specific data
        load(['Data_Class_',num2str(class),'_Subject_',num2str(sub),'.mat'])
        % Can replace with a more typical Cross validation procedure by
        % dividing into independent 5 folds. Doing random sampling 5 set 
        % method since number of grouping is not balanced and to ensure 
        % no additional imbalance is created by the random folds
        
        for iter = 1 : 5 
            
            % Get location of data corresponding to each lab
            loc0 = find(labels==0);
            loc1 = find(labels==1);
            
            % Get unique trial IDs
            class0=unique(identifier_val(loc0,1));
            class1=unique(identifier_val(loc1,1));
            
            % Randomize the trial order per class
            class1=class1(randperm(length(class1)));
            class0=class0(randperm(length(class0)));
            
            % Divide into training and test set            
            Test=[class0(1:4);class1(1:4)]; % ~20 percent trials for for testing (8/40)
            Train=[class0(5:end);class1(5:end)]; % ~20 percent trials for for training
            
            % Identify locations in the feature set and labels to divide
            % into training and test set based on the trial identifier
            
            train_loc   = find(ismember(identifier_val(:,1),Train));
            train_loc   = train_loc(randperm(length(train_loc)));
            test_loc    = find(ismember(identifier_val(:,1),Test));
            test_loc    = test_loc(randperm(length(test_loc)));
            
           
            X=Feature(:,:);
            
            % Do PCA to reduce collinearity
            [coeff,score,latent,tsquared,explained,mu] = pca(X);
            X=score(:,find(cumsum(explained)<96));
            
            
            X_train=X(train_loc,:);            
            Y_train=labels(train_loc);
            
            X_test=X(test_loc,:);
            Y_test=labels(test_loc);
            
%             treeStump = templateTree('MaxNumSplits',10);
            treeStump = templateTree('MaxNumSplits',1); % The ensemble model is not hyperparameter optimized
            ens = fitcensemble(X_train,Y_train,'Method','Bag','Learners',treeStump);
            
            
            
            %% Performance metrics
            label_pred = predict(ens,X_test);
            
            % Test accuracy
            Acc(sub,iter,class)=1-sum(abs(label_pred-Y_test))/length(Y_test);            
            % Confusion matrix
            [confMat,order] = confusionmat(label_pred,Y_test);  
            end
    end
end
%%
disp('training done')
mean(squeeze(mean(Acc,2)))
max(squeeze(mean(Acc,2)))

