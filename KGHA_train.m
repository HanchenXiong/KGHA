function [dual_Alpha]=KGHA_train(v_train_Kernel, v_train_Label,p,lambda,c)
num_fea=size(v_train_Kernel,2);                % number of features
[num_ins, num_lab]=size(v_train_Label);        % number of instances and number of labels 

dual_Alpha=cell(1,num_fea+1);                  % each cell-unit is a alpha vector for a feature 


% construct multi-label kernel
mean_Label=mean(v_train_Label);
v_train_Label=v_train_Label-repmat(mean_Label,num_ins,1);        % centralize multi-label outputs   

output_kernel=v_train_Label*v_train_Label';                      

%randomly initialize the p-dimensional projections
X=randn(num_ins, p);
old_X=X;

c_step=0;
n_step=100;  % maximum number of iterations in solving ALS  

while c_step<n_step
    c_step=c_step+1;

    % ----update alpha
    for i=1:num_fea
        dual_Alpha{1,i}=pinv(v_train_Kernel{1,i}+c*eye(num_ins,num_ins))*X;
    end;
    dual_Alpha{1,num_fea+1}=pinv(output_kernel+c*eye(num_ins,num_ins))*X;


    % ----update X
    X=0;  
    for i=1:num_fea
        X=X+lambda*v_train_Kernel{1,i}*dual_Alpha{1,i};
    end;
    X=X+output_kernel*dual_Alpha{1,num_fea+1};

    X=X/(lambda*num_fea+1);


    % ---- normalize X
    X=Gram_Schmidt_process(X);
    
    % if the projection does not change very much  
    if norm(X-old_X)/(num_ins*p)<0.0001
        % before stoping iteration, finding the optimal dual variables corresponding to newest X     
        for i=1:num_fea
            dual_Alpha{1,i}=pinv(v_train_Kernel{1,i}+c*eye(num_ins,num_ins))*X;
        end;
        dual_Alpha{1,num_fea+1}=pinv(output_kernel+c*eye(num_ins,num_ins))*X;
 
        % now go out
        break;
    end; 
    old_X=X;
end;


