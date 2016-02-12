function []=Bio_Experiments(pre_domain)

Kernel=load([pre_domain,'_Kernel.mat']);
Kernel=Kernel.([pre_domain,'_Kernel']);

Label=load([pre_domain,'_Label.mat']);
Label=Label.([pre_domain,'_Label']);

num_fea=size(Kernel,2);
[num_ins,num_lab]=size(Label);

if (pre_domain=='Human')
   num_sel=floor(num_ins*0.6);
   num_tst=floor(num_ins*0.4);
elseif (pre_domain=='Yeast')
    num_sel=floor(num_ins*0.8);
    num_tst=floor(num_ins*0.2);
end;


method='R';
num_trial=3;
ratio=[0.2, 0.4, 0.5, 0.6, 0.8];
num_ratio=size(ratio,2);
  
all_performance=zeros(3,num_ratio);

for j=1:num_trial
    fprintf('The %d -th trial out of %d on %s dataset......\n', j, num_trial, pre_domain);
    random_idx=randperm(num_ins);
    sel_idx=random_idx(1:num_sel);
    Kernel_sel=cell(1,num_fea);
    Label_sel=Label(sel_idx,:);
    
    tst_idx=random_idx(num_sel+1:num_sel+num_tst);
    Kernel_cro=cell(1,num_fea);
    Label_tst=Label(tst_idx,:);
    
    
    for i=1:num_fea
        Kernel_sel{1,i}=Kernel{1,i}(sel_idx,sel_idx);
        Kernel_cro{1,i}=Kernel{1,i}(tst_idx,sel_idx);
    end;
    Label_sel=Label(sel_idx,:);
    
  
    for i=1:num_ratio
        p=round(num_lab*ratio(i))
        [o_lambda, o_c] = KGHA (Kernel_sel, Label_sel, p, method)
        [o_Alpha]=KGHA_train(Kernel_sel, Label_sel, p, o_lambda, o_c);
        [performance]=Eval_test(o_Alpha,Kernel_cro,Label_sel,Label_tst, method)
        all_performance(j,i)=performance; 
    end;
end;

save(['KGHA_', pre_domain, '_performance.mat'],'all_performance');
