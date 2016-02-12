function [performance]=Eval_test(dual_Alpha,v_cross_Kernel,v_train_Label, v_test_Label,method)
num_fea=size(v_cross_Kernel,2);
[num_ins,num_lab]=size(v_test_Label);


% predict the label score
X=0;
for i=1:num_fea
      X=X+v_cross_Kernel{1,i}*dual_Alpha{1,i};
end;
X=X/num_fea;


pred_score=X*(dual_Alpha{1,num_fea+1}'*v_train_Label)*pinv(v_train_Label'*dual_Alpha{1,num_fea+1}*dual_Alpha{1,num_fea+1}'*v_train_Label);


%----------------------- evluate with 1-(R)ankingLoss or Macro(F)1---------------------
if method=='R'
     performance=0; 
     for i=1:num_ins 
         performance_i=0;

         true_idx=find(v_test_Label(i,:)==1);   % annoated labels in groud truth
         num_true_idx=size(true_idx,2);
 
       
         pred_score_i=pred_score(i,:);
         pred_score_i_r=pred_score_i;
         pred_score_i_r(true_idx)=[];      % remove true label indices from predicted scores

         for j=1:size(true_idx,1)
             wrong_ordered_idx=find(pred_score_i_r>pred_score_i(true_idx(j))); 
             num_wrong_ordered=size(wrong_ordered_idx,2);
             performance_i=performance_i+num_wrong_ordered;
         end;
         
         performance=performance+performance_i/(num_true_idx*(num_lab-num_true_idx));
     end;
     performance=1-performance/num_ins; 

     
     %-----------------------Precision, Recall and F1 --------------------------------------
elseif method=='F'
    l=5;      % the desired number of labels
    binary_pred=-ones(num_ins,num_lab);
    for i=1:num_ins
        pred_score_i=pred_score(i,:);   
        [ranked_pred_score_i, ranked_idx]=sort(pred_score_i,'descend');
        top_idx=ranked_idx(1:l);
        binary_pred(i,top_idx)=1;
    end;
    
    % convert [-1,1] to [0,1]
    binary_pred=(binary_pred+1)/2;
    v_test_Label=(v_test_Label+1)/2;

    true_positive=binary_pred.*v_test_Label;
     
    precision=0;
    recall=0;

    [x,y,v]=find(true_positive==1);
    num_true_positive=size(x,1);
    
    [x,y,v]=find(binary_pred==1);
    pred_positive=size(x,1);

    [x,y,v]=find(v_test_Label==1);
    real_positive=size(x,1);


    precision=precision+num_true_positive/pred_positive
    recall=recall+num_true_positive/real_positive
    F1=2*precision*recall/(precision+recall)

   % for i=1:num_lab
   %     i_true_positive=size(find(true_positive(:,i)==1),1)
   %     pred_positive=size(find(binary_pred(:,i)==1),1)
   %     real_positive=size(find(v_test_Label(:,i)==1),1)

   %     if pred_positive==0
   %         precision=0;
   %     end;

   %     if real_positive==0
   %         recall=0;
   %     end;

   %     if  (pred_positive~=0)&& (real_positive~=0)
   %         precision=precision+i_true_positive/pred_positive;
   %         recall=recall+i_true_positive/real_positive;
   %     end;
   % end;

   % precision=precision/num_lab;    % average precision
   % recall=recall/num_lab;          % average recall
   % F1=2*precision*recall/(precision+recall);

    performance.precision=precision;
    performance.recall=recall;
    performance.F1=F1;
end;

