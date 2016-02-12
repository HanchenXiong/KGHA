function []=Image_Experiments(pre_domain, kernel_type)

train_Kernel=load([pre_domain,'_', kernel_type, '_train_Kernels.mat']);
train_Kernel=train_Kernel.train_Kernel;

train_Label=load([pre_domain, '_train_Label.mat']);
train_Label=train_Label.train_Label;

cross_Kernel=load([pre_domain, '_', kernel_type, '_cross_Kernels.mat']);
cross_Kernel=cross_Kernel.cross_Kernel;

test_Label=load([pre_domain,'_test_Label.mat']);
test_Label=test_Label.test_Label;


[num_tra_ins,num_lab]=size(train_Label);
[num_tst_ins,num_lab]=size(test_Label);

ratio=[0.2, 0.4, 0.5, 0.6, 0.8];
num_ratio=size(ratio,2);


method='F';
  
all_performance=zeros(1,num_ratio);

for i=1:num_ratio
    p=round(num_lab*ratio(i))
    [o_lambda, o_c] = KGHA (train_Kernel, train_Label, p, method)
    [o_Alpha]=KGHA_train(train_Kernel, train_Label, p, o_lambda, o_c);
    [performance]=Eval_test(o_Alpha,cross_Kernel,train_Labe,test_Label, method)
    all_performance(j,i)=performance; 
end;

save(['KGHA_', pre_domain, '_performance.mat'],'all_performance');
