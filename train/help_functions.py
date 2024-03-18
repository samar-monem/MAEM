!pip install xlrd==2.0.1 
!pip install Pandas==1.3.5
!pip install  gdown
!pip install deepchem
!pip install pysmiles
!pip install numpy==1.24.3



import numpy as np
import pandas as pd
 
def normalize1(X, means1=None, std1=None, means2=None, std2=None, feat_filt=None, norm='tanh_norm'):
    if std1 is None:
        std1 = np.nanstd(X, axis=0)
    if feat_filt is None:
        feat_filt = std1!=0
    X = X[:,feat_filt]
    X = np.ascontiguousarray(X)
    if means1 is None:
        means1 = np.mean(X, axis=0)
    X = (X-means1)/std1[feat_filt]
    if norm == 'norm':
        return(X, means1, std1, feat_filt)
    elif norm == 'tanh':
        return(np.tanh(X), means1, std1, feat_filt)
    elif norm == 'tanh_norm':
        X = np.tanh(X)
        if means2 is None:
            means2 = np.mean(X, axis=0)
        if std2 is None:
            std2 = np.std(X, axis=0)
        X = (X-means2)/std2
        X[:,std2==0]=0
        return(X, means1, std1, means2, std2, feat_filt) 


def repeat_smiles1(data_to_repeat,unique_drugs,feature,unique_cell,finger_drug,graph,target,copy,mutation,protematics):
#   unique_finger_feature=finger_drug[:,1:]
#   unique_finger_name=finger_drug[:,0]
  unique_feature=unique_cell[:,1:]
  unique_name=unique_cell[:,0]
  d1=data_to_repeat[:,0]
  d2=data_to_repeat[:,1]
  c=data_to_repeat[:,2]
  unique_drugs=feature[:,0]
  feature=feature[:,1:]  
  f_drug1=[]
  f_drug2=[]
  feature_cell=[]
  finger_d1=[]
  finger_d2=[]
  graph1=[]
  graph2=[]
  target1=[]
  target2=[]
  ccopy=[]
  cmutation=[]
  cprotematics=[]

  for i in range(len(d1)):
    n1=d1[i]
    n2=d2[i]
    cc=c[i]
    k1= [m for m, v in enumerate(unique_drugs) if n1 in v]
    k2=[m for m, v in enumerate(unique_drugs) if n2 in v]
    cc1=[m for m, v in enumerate(unique_name) if cc in v]
#     r1= [m for m, v in enumerate(unique_redkit_name) if n1 in v]
#     r2=[m for m, v in enumerate(unique_redkit_name) if n2 in v]
    f_drug1.append(feature[k1[0]])
    f_drug2.append(feature[k2[0]])
    graph1.append(graph[k1[0]])
    graph2.append(graph[k2[0]])
    finger_d1.append(finger_drug[k1[0]])
    finger_d2.append(finger_drug[k2[0]])
    target1.append(target[k1[0]])
    target2.append(target[k2[0]])
    
    feature_cell.append(unique_feature[cc1[0]])
    ccopy.append(copy[cc1[0]])
    cmutation.append(mutation[cc1[0]])
    cprotematics.append(protematics[cc1[0]])

  return f_drug1,f_drug2,feature_cell,finger_d1,finger_d2,graph1,graph2,target1,target2,ccopy,cmutation,cprotematics




def train_test_input(f_drug1,f_drug2,cell_line,index_train,index_test,synery,class1,finger_d1,finger_d2,graph1,graph2,target1,target2,copy,mutation,protematics):
  train_f_drug1=[]
  train_f_drug2=[]
  train_cell_line=[]
  train_synergy=[]
  train_class=[]
  train_finger_d1=[]
  train_finger_d2=[]
  test_f_drug1=[]
  test_f_drug2=[]
  test_cell_line=[]
  test_synergy=[]
  test_class=[]
  test_finger_d1=[]
  test_finger_d2=[]
  train_graph1=[]
  test_graph1=[]
  train_graph2=[]
  test_graph2=[]
    
  train_target1=[]
  train_target2=[]
  train_copy=[]
  train_mutation=[]
  train_protematics=[]
  test_target1=[]
  test_target2=[]
  test_copy=[]
  test_mutation=[]
  test_protematics=[]
  for i in range(len(index_train)):
      
      train_f_drug1.append(f_drug1[index_train[i]])
      train_f_drug2.append(f_drug2[index_train[i]])
      train_cell_line.append(cell_line[index_train[i]])
      train_synergy.append(synergy[index_train[i]])
      train_class.append(class1[index_train[i]])
      train_finger_d1.append(finger_d1[index_train[i]])
      train_finger_d2.append(finger_d2[index_train[i]])
      train_graph1.append(graph1[index_train[i]])
      train_graph2.append(graph2[index_train[i]])
     
      train_target1.append(target1[index_train[i]]) 
      train_target2.append(target2[index_train[i]])
      train_copy.append(copy[index_train[i]])
      train_mutation.append(mutation[index_train[i]])
      train_protematics.append(protematics[index_train[i]])
 
  for ii in range(len(index_test)):
      
      test_f_drug1.append(f_drug1[index_test[ii]])
      test_f_drug2.append(f_drug2[index_test[ii]])
      test_cell_line.append(cell_line[index_test[ii]])
      test_synergy.append(synergy[index_test[ii]])
      test_class.append(class1[index_test[ii]])
      test_finger_d1.append(finger_d1[index_test[ii]])
      test_finger_d2.append(finger_d2[index_test[ii]])
      test_graph1.append(graph1[index_test[ii]])
      test_graph2.append(graph2[index_test[ii]])
        
      test_target1.append(target1[index_test[ii]])
      test_target2.append(target2[index_test[ii]])
      test_copy.append(copy[index_test[ii]])
      test_mutation.append(mutation[index_test[ii]])
      test_protematics.append(protematics[index_test[ii]])

  return train_f_drug1,train_f_drug2,train_cell_line,test_f_drug1,test_f_drug2,test_cell_line,train_synergy,train_class,test_synergy,test_class,train_finger_d1,train_finger_d2,test_finger_d1,test_finger_d2,train_graph1,test_graph1,train_graph2,test_graph2,train_target1,train_target2,train_copy,train_mutation,train_protematics,test_target1,test_target2,test_copy,test_mutation,test_protematics



def preprocess(index_train,index_test):
    index_train1=[]
    index_test1=[]
    index_train2=(index_train)[0]
    index_test2=(index_test)[0]
    for i in range(len((index_train2))):
        index_train1.append((index_train2[i]))
        
    for ii in range(len(index_test2)):
        index_test1.append((index_test2[ii]))
        
    return index_train1,index_test1

def get_data_me2(s):

    !gdown 1C7Z2ziPdQVzH3omIdIfyJa7VUmog4IIk
    labels = pd.read_csv('oneil.csv', index_col=0) 
    
    h=len(np.array(labels))
    #labels are duplicated for the two different ways of ordering in the data
    labels = pd.concat([labels, labels]) 
    
    test_fold =s
   
    idx_train = np.where(labels['fold']!=test_fold)
    

    idx_test = np.where(labels['fold']==test_fold)
#     
#    
    return idx_train,idx_test





def convert_tobin(cc):
    cb=[]
    for i in range(len(cc)):
        if(cc[i]>=0.5):
            cb.append(1)
        else:
            cb.append(0)
    return cb


def norm1(train_cell_line,test_cell_line,norm="tanh_norm"):
# norm = "norm"
    if norm == "tanh_norm":
        train_cell_line, mean, std, mean2, std2, feat_filt = normalize1(train_cell_line, norm=norm)
        test_cell_line, mean, std, mean2, std2, feat_filt = normalize1(test_cell_line, mean, std, mean2, std2, 
                                                              feat_filt=feat_filt, norm=norm)
    else:
        train_cell_line, mean, std, feat_filt = normalize1(train_cell_line, norm=norm)
        test_cell_line, mean, std, feat_filt = normalize1(test_cell_line, mean, std, feat_filt=feat_filt, norm=norm)
    
    return train_cell_line,test_cell_line
