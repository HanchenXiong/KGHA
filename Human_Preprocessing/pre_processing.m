human_and_go=load('human_and_go.mat');

raw_Kernel=human_and_go.networks;
raw_Label=human_and_go.GO;

raw_Label=raw_Label.data;
[num_ins, num_lab]=size(raw_Label);



% find non_zeros indices in the multi-label output matrix 
[idx_i, idx_j, v]=find(raw_Label);

% select function indieces with at least 30 instances and at most 100 instances
sel_fidx=[];
for j=1:num_lab
   num_fj=size(find(idx_j==j),1) ;  % number of instances which have function j

   if (num_fj>30) && (num_fj<=100)
      sel_fidx=[sel_fidx;j];        % sel_fidx: selected function indices which have at least 30 instances and at most 100 instances
   end;    
end;

num_sel_f=size(sel_fidx,1);

sel_pidx=[];
for i=1:num_sel_f
    idx=find(idx_j==sel_fidx(i));       % the paris which has sel_fidx(i)
    sel_pidx=[sel_pidx; idx_i(idx)];   
end;
sel_pidx=unique(sel_pidx);              % the unique protein indices 

%num_sel=size(sel_pidx,1) 


% generate appropriate output labels 
Human_Label=raw_Label(sel_pidx, sel_fidx);
Human_Label=full(Human_Label);          % convert sparse martix to full matrix
Human_Label=2*Human_Label-1;            % convert {0,1} to {-1,1}
save('Human_Label.mat', 'Human_Label'); 
    


% generate appropirate input kernels 
num_fea=size(raw_Kernel,2);
Human_Kernel=cell(1,num_fea); 
for i=1:num_fea
    Human_Kernel{1,i}=raw_Kernel{1,i}.data;
end;
save('Human_Kernel.mat', 'Human_Kernel'); 
