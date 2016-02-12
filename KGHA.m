function [o_lambda, o_c] = KGHA (Kernel, Label, p, method)

% input:   Kernel --> the kernel matrices of training instances with (J-1) heterogeneous features
%                    a cell structure with (J-1) matrices   
%
%          Label  --> the annotated lable for all instances 
%                    a M by d_J binary matrix, where d_J is the number of lables  
%          
%           p     --> the dimensionality of latent space
%         method  --> the evaluation method, can be 'F'(1 measure) or 'R'(ankingLoss)
%
% output:  o_lambda, o_c  --> optimal hyperparameters 


%-------------------------data infor extraction-----------------------------
% detect the number of input features
num_fea=size(Kernel,2);                % number of features
[num_ins, num_lab]=size(Label);        % number of instances and number of labels 


%------------------------heyper-parameters----------------------------------
lambda=[0.00001, 0.0001, 0.001, 0.01]; 
     c=[0.00001, 0.0001, 0.001, 0.01];
%lambda=[0.001, 0.01]; 
%     c=[0.0001, 0.001];
%------------------------4-fold cross-validation----------------------------
% set 4 folds of instances
rand_idx=randperm(num_ins);    % reshuffel all indices

percent=p/(num_ins*0.2);  % percentage of data for cross validation since it is a very expensive process, however, the number of instances in one fold should be larger than the number of lables (because Gram-Schmidt process). 

if percent<=0.2
    percent=0.2
end;

fold_1_idx=rand_idx(1:floor(num_ins*0.25*percent));
fold_2_idx=rand_idx(floor(num_ins*0.25*percent)+1:floor(num_ins*0.5*percent));
fold_3_idx=rand_idx(floor(num_ins*0.5*percent)+1:floor(num_ins*0.75*percent));
fold_4_idx=rand_idx(floor(num_ins*0.75*percent)+1:floor(num_ins*percent));

four_folds=cell(1,4);
four_folds{1,1}=fold_1_idx;
four_folds{1,2}=fold_2_idx;
four_folds{1,3}=fold_3_idx;
four_folds{1,4}=fold_4_idx;


n_trial=size(lambda,2)*size(c,2);  % the whole number trial for hyper-parameter tunning
c_trial=0;                         % the current number trial 

trial_results=zeros(size(lambda,2),size(c,2));   % store the performances of all trials 
trial_alphas=cell(size(lambda,2),size(c,2));     % store the learned alpha of all trials


for i=1:size(lambda,2)
    for j=1:size(c,2) 
        ave_performance=0;
        c_trial=c_trial+1;
        fprintf('The %d -th trial out of %d for cross-validation........\n', c_trial, n_trial); 

        for k=1:4    % 4-fold cross validation
            % use 3 for validation_training data and the remaining one for validation_test data 
            v_test_set_idx=four_folds{1,k};
            % actually test_kernel is not necessary for test:   v_test_Kernel=cell(1,num_fea); 
            v_cross_Kernel=cell(1,num_fea);
           
            v_train_set=[1,2,3,4];
            v_train_set(k)=[];
            v_train_set_idx=[four_folds{1,v_train_set(1)},four_folds{1,v_train_set(2)},four_folds{1,v_train_set(3)}]; 
            v_train_Kernel=cell(1,num_fea);
            
         
            % copy all num_fea kernels
            for tt=1:num_fea
            % actually test_kernel is not necessary for test:   v_test_Kernel{1,tt}=Kernel{1,tt}(v_test_set_idx,v_test_set_idx);
                v_cross_Kernel{1,tt}=Kernel{1,tt}(v_test_set_idx,v_train_set_idx);
                v_train_Kernel{1,tt}=Kernel{1,tt}(v_train_set_idx,v_train_set_idx);
            end;
           
            % copy labels
            v_test_Label=Label(v_test_set_idx,:);
            v_train_Label=Label(v_train_set_idx,:);

            % training and testing for validation
            [dual_Alpha]=KGHA_train(v_train_Kernel, v_train_Label, p, lambda(i), c(j));
            
            if method=='R'
                [performance]=Eval_test(dual_Alpha,v_cross_Kernel,v_train_Label, v_test_Label,method);
                ave_performance=ave_performance+performance;
            elseif method=='F'
                [performance]=Eval_test(dual_Alpha,v_cross_Kernel,v_train_Label, v_test_Label,method);
                ave_performance=ave_performance+performance.F1;
             end;
        end; 
        trial_results(i,j)=ave_performance/4;
        trial_alphas{i,j}=dual_Alpha;
    end;
end;


%-------------------- find optimal hyperparameters--------------
trial_results

[max_value_set,lambda_idx]=max(trial_results);     % e.g.  max_value_set: /*  0.65, 0.72, 0.78,  0.77, 0.71  */,   lambda_idx: /*    3  5  6  3  1    */    
[max_value, c_idx]=max(max_value_set);             % e.g.  max_value: 0.78,  c_idx: 3 

o_c=c(c_idx(1));                                   % e.g.  o_c=c(3)   
o_lambda=lambda(lambda_idx(c_idx(1)));             % e.g.  o_lambda=lambda(lambda_idx(3))=lambda(6);         
o_Alpha=trial_alphas{lambda_idx,c_idx};            



