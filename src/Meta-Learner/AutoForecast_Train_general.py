import pandas as pd
import os
import simplejson as json
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import sklearn
from sklearn import preprocessing
from numpy import inf
from sklearn.decomposition import PCA
import time

# linear regression for multioutput regression
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge 


## Calculating Average. value of a list
def Average_val(lst):
    return sum(lst) / len(lst)

## Normalizing feature vectors
def Normalize_feature(feat_df, PCA_option):  
    if PCA_option == 'No':
        x = feat_df #.values 
    else:
        x = feat_df    
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    res = pd.DataFrame(x_scaled)
    
    return res

### Check about empty files of a directory 
def Check_Empty_Files(dirName):
    
    # Create a List    
    listOfEmptyDirs = list()

    # Iterate over the directory tree and check if directory is empty.
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        if len(dirnames) == 0 and len(filenames) == 0 :
            listOfEmptyDirs.append(dirpath)

### Get Meta-Features matrix for the training and testing 
def Get_Meta_Features (meta_feat_dir, data_list, data_type):
    
    df_meta_feat = pd.DataFrame()
    
    if data_type == 'Uni-var':
        
        for file_name in data_list:
            print(file_name)
            df = pd.read_csv(meta_feat_dir + file_name +'.csv', error_bad_lines=False, header  = None)
            last_row = df.tail(1)
            df_meta_feat = df_meta_feat.append(last_row, ignore_index=True)
            df_meta_feat = df_meta_feat.fillna(0)
            
    
    return df_meta_feat

    '''
               else:      
                   for file_name in all_datasets_list:
                      cnt = 0
                      results_vec = []
                      if '.DS_Store' not in file_name:
                          cnt += 1
                          all_models_list = os.listdir(win_res_dir_name + '/' + dir_name + '/' + file_name) 
                                
                      
                          for model_name in all_models_list:
                              df = pd.read_csv(win_res_dir_name + '/' + dir_name+'/'+file_name+'/'+model_name, error_bad_lines=False, header  = None)
                              results_vec.append(df.iloc[0,-1])
                      
                          best_model_dataset_index = results_vec.index(min(results_vec))
                          print(file_name + ':' + arr_models[best_model_dataset_index + 1])
                      
                          ## Append Best Model for each dataset variable to the vector
                          a.append(best_model_dataset_index + 1)
                      
                          results_vec.insert(0,dir_name+'_'+file_name)
                      
                          numpy_perf_vec = np.array(results_vec).reshape((1, len(all_models_list)+1))    
                          numpy_perf_list = numpy_perf_vec.tolist()
                      
                          wtr_perf.writerow (numpy_perf_list)
    
    return a    
'''    
   
### Get the best model for specific dataset (Used for ALgros baseline)
def Get_All_Model_Dataset(data_name, win_res_dir_name, data_type):
    
    arr_models = os.listdir(win_res_dir_name + '/' + data_name)
    
    all_models_list = os.listdir(win_res_dir_name + '/' + data_name) 
     
    results_vec_mse = []
    results_vec_mape = []
    results_vec_smape = []
    for model_name in all_models_list:
        df = pd.read_csv(win_res_dir_name + '/' + data_name+'/'+ model_name, error_bad_lines=False, header  = None)
        results_vec_mse.append(df.iloc[0,-1])
        results_vec_mape.append(df.iloc[1,-1])
        results_vec_smape.append(df.iloc[2,-1]) 
    
    
    return results_vec_mse, results_vec_mape, results_vec_smape, arr_models
        
### (a) Gloal Best Implementation
def Global_Best (dir_list, data_list, data_type):
    Models_Array_mse = [] 
    Models_Array_mape = []
    Models_Array_smape = []  
    for data_name in data_list:
        for win_ind in os.listdir(dir_list):
            if '.DS_Store' not in win_ind:
                    print(data_name)
                    ##### Get Performance Matrix and Best Model Array for both Multi-variate and Uni-variate Datasets with a specific window
                    if data_type == 'Uni-var':
                            results_vec_mse = []
                            results_vec_mape = []
                            results_vec_smape = []
                            
                            all_models_list = os.listdir(dir_list + win_ind + '/' + data_name)  
                            for model_name in all_models_list:
                                df = pd.read_csv(dir_list + win_ind + '/' + data_name + '/' + model_name, error_bad_lines=False, header  = None)
                                results_vec_mse.append(df.iloc[0,-1])
                                results_vec_mape.append(df.iloc[1,-1])
                                results_vec_smape.append(df.iloc[2,-1])
                
                            best_model_dataset_index_mse = results_vec_mse.index(min(results_vec_mse))
                            best_model_dataset_index_mape = results_vec_mape.index(min(results_vec_mape))
                            best_model_dataset_index_smape = results_vec_smape.index(min(results_vec_smape))
                            
                            
                            ## Append Best Model for each dataset variable to the vector
                            Models_Array_mse.append(best_model_dataset_index_mse)
                            Models_Array_mape.append(best_model_dataset_index_mape)
                            Models_Array_smape.append(best_model_dataset_index_smape)

           
    d_mse = defaultdict(int)
    for i in Models_Array_mse:
        d_mse[i] += 1
    result_mse = max(d_mse.items(), key=lambda x: x[1])
    
    d_mape = defaultdict(int)
    for i in Models_Array_mape:
        d_mape[i] += 1
    result_mape = max(d_mape.items(), key=lambda x: x[1])
    
    d_smape = defaultdict(int)
    for i in Models_Array_smape:
        d_smape[i] += 1
    result_smape = max(d_smape.items(), key=lambda x: x[1])
    
    return all_models_list[result_mse[0]], all_models_list[result_mape[0]], all_models_list[result_smape[0]]
    
### Get Average Performance of Models across dataset cluster (Used for ISAC baseline)
def Get_Best_Avg_Model_Dataset(data_cluster, win_res_dir_name, data_type):
    
    results_vec_mse_avg = [0] * 322
    results_vec_mape_avg = [0] * 322
    results_vec_smape_avg = [0] * 322
    
    for data_name in data_cluster:
        
        results_vec_mse = []
        results_vec_mape = []
        results_vec_smape = []
        
        arr_models = os.listdir(win_res_dir_name + '/' + data_name)
        all_models_list = os.listdir(win_res_dir_name + '/' + data_name) 
    
        for model_name in all_models_list:
            df = pd.read_csv(win_res_dir_name + '/' + data_name+'/'+ model_name, error_bad_lines=False, header  = None)
            
            results_vec_mse.append(df.iloc[0,-1])
            results_vec_mape.append(df.iloc[1,-1])
            results_vec_smape.append(df.iloc[2,-1]) 
        
        results_vec_mse_avg  += results_vec_mse
        results_vec_mape_avg  += results_vec_mape
        results_vec_smape_avg  += results_vec_smape    
    
    best_model_dataset_index_mse_avg = results_vec_mse_avg.index(min(results_vec_mse_avg))
    best_model_dataset_index_mape_avg = results_vec_mape_avg.index(min(results_vec_mape_avg))
    best_model_dataset_index_smape_avg = results_vec_smape_avg.index(min(results_vec_smape_avg))
    
    
    return arr_models[best_model_dataset_index_mse_avg], arr_models[best_model_dataset_index_mape_avg], arr_models[best_model_dataset_index_smape_avg] 

### Get the best model for specific dataset (Used for ALgros baseline)
def Get_Best_Model_Dataset(data_name, win_res_dir_name, data_type):
    
    arr_models = os.listdir(win_res_dir_name + '/' + data_name)
    arr_models.insert(0,'Dataset')
    
    all_models_list = os.listdir(win_res_dir_name + '/' + data_name) 
    
     
    results_vec_mse = []
    results_vec_mape = []
    results_vec_smape = []
    for model_name in all_models_list:
        df = pd.read_csv(win_res_dir_name + '/' + data_name+'/'+ model_name, error_bad_lines=False, header  = None)
        results_vec_mse.append(df.iloc[0,-1])
        results_vec_mape.append(df.iloc[1,-1])
        results_vec_smape.append(df.iloc[2,-1]) 
    
    
    best_model_dataset_index_mse = results_vec_mse.index(min(results_vec_mse))
    best_model_dataset_index_mape = results_vec_mape.index(min(results_vec_mape))
    best_model_dataset_index_smape = results_vec_smape.index(min(results_vec_smape))
    
    
    return arr_models[best_model_dataset_index_mse + 1], arr_models[best_model_dataset_index_mape + 1], arr_models[best_model_dataset_index_smape + 1]
    
#### Get Model Files list from a dataset directory
def Get_Model_Files_List(win_res_dir_name, data_type):
    
    dir_datasets_name = os.listdir(win_res_dir_name)   
    for dir_name in dir_datasets_name:
     
           if '.DS_Store' in dir_name:
               continue
           
           if data_type == 'Uni-var':
                all_models_list = os.listdir(win_res_dir_name+'/'+dir_name)
           
           else:
               all_datasets_list = os.listdir(win_res_dir_name+'/'+dir_name) 
           
               for file_name in all_datasets_list:
                  if '.DS_Store' not in file_name:
                      all_models_list = os.listdir(win_res_dir_name+'/'+dir_name+'/'+file_name) 
                      
                         
    return all_models_list 

#### Draw and Save Histogram for the best models
def Histogram_plot_save(window_best_models_arr):
    
    _ = plt.hist(window_best_models_arr, bins= 'auto', density = True)  # arguments are passed to np.histogram
    plt.title("Histogram of Best Models for Training Dataset", fontsize=12)
    plt.xlabel("Forecasting Model Index", fontsize=12)
    plt.ylabel("Probability of being Best Model", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=12)
    plt.savefig('hist_best_models.eps', format='eps')
    plt.show()
    
def chunks(l, n):
        n = max(1, n)
        return [l[i:i+n] for i in range(0, len(l), n)]

### Get Difference between two lists
def Diff(li1, li2): 
        return (list(set(li1) - set(li2)))

#### Divide the datasets into 5 folds for training and testing 
def Divide_Dataset_Folds(n_folds, datasets_dir_list):
    
    train_folds_list = []
    test_folds_list = []

    length = int(len(datasets_dir_list)/ n_folds) #length of each fold
    folds = []
    for i in range(n_folds-1):
        folds += [datasets_dir_list[i*length:(i+1)*length]]
    folds += [datasets_dir_list[4*length:len(datasets_dir_list)]]
    
    print(folds)
    
    for fold in folds:        
        train_folds_list.append(Diff(datasets_dir_list, fold))
        test_folds_list.append(fold)


    return train_folds_list, test_folds_list    
        

if __name__ == '__main__':
    
    n_folds = 5
    PCA_option = 'Yes' #Yes
    n_components = 5 ## If PCA_option is Yes, the n_components
    
    ## Folds for MV datasets
    mv_dir = 'results_all_mv/Multi-variate_first_win/'
    dir_mv_list = os.listdir(mv_dir)
    dir_mv_list.remove('.DS_Store')
    train_folds_list_mv, test_folds_list_mv = Divide_Dataset_Folds(n_folds, dir_mv_list) ## Division of folds by dataset names
    
    
    ## Folds for UV datasets
    #### Directories for perfromances results for all other windows
    
    uv_dir_1 = 'results_all_mv/Multi-variate_first_win/'
    uv_dir_2 = 'results_all_mv/Multi-variate_second_win'
    uv_dir_3 = 'results_all_mv/Multi-variate_third_win'
    dir_uv_list = os.listdir(uv_dir_1)
    dir_uv_list.remove('.DS_Store')
    train_folds_list_uv, test_folds_list_uv = Divide_Dataset_Folds(n_folds, dir_uv_list) ## Division of folds by dataset names
      
  
    #### Go across the folds and evaluate the different baselines and then average
    meta_feat_dir_1 = 'Meta-Features/Meta-Feat_first_win/Multi_Variate_Real_first_win/'
    meta_feat_dir_2 = 'Meta-Features/Meta-Feat_sec_win/Multi_Variate_Real_sec_win/'
    meta_feat_dir_3 = 'Meta-Features/Meta-Feat_third_win/Multi_Variate_Real_third_win/'
    
    '''     
    uv_dir_1 = 'results_all_uv/Uni-variate_first_win/' 
    uv_dir_2 = 'results_all_uv/Uni-variate_second_win'
    uv_dir_3 = 'results_all_uv/Uni-variate_third_win'
    dir_uv_list = os.listdir(uv_dir_1)
    dir_uv_list.remove('.DS_Store')
    train_folds_list_uv, test_folds_list_uv = Divide_Dataset_Folds(n_folds, dir_uv_list) ## Division of folds by dataset names
    
    #print('UV<<<<') #print(train_folds_list_uv) #print(test_folds_list_uv)
    
    
    #### Go across the folds and evaluate the different baselines and then average
    meta_feat_dir_1 = 'Meta-Features/Meta-Feat_first_win/Uni-Variate_first_win/'
    meta_feat_dir_2 = 'Meta-Features/Meta-Feat_sec_win/Uni-Variate_sec_win/'
    meta_feat_dir_3 = 'Meta-Features/Meta-Feat_third_win/Uni-Variate_third_win/'
    '''
       
    results_vec_gen_mse = []
    results_vec_gen_mape = []
    results_vec_gen_smape =[]
    cnt_k_all = 0
    
    train_time_vec = []
    for i in range(0, n_folds):
        
        test_Y_mse = pd.DataFrame()
        #test_Y_mape = pd.DataFrame()
        #test_Y_smape = pd.DataFrame()
        
        
        train_Y_mse = pd.DataFrame()
        #train_Y_mape = pd.DataFrame()
        #train_Y_smape = pd.DataFrame()
        
        
        #train_Y_mape = []; train_Y_smape = []
        
        ### Extract Meta-Features
        ## (1) Train Meta-features
        meta_features_train_orig_1 = Get_Meta_Features (meta_feat_dir_1, train_folds_list_uv[i], 'Uni-var')
        meta_features_train_orig_2 = Get_Meta_Features (meta_feat_dir_2, train_folds_list_uv[i], 'Uni-var')
        meta_features_train_orig_3 = Get_Meta_Features (meta_feat_dir_3, train_folds_list_uv[i], 'Uni-var')
        
        meta_features_train_orig_par = meta_features_train_orig_1.append(meta_features_train_orig_2, ignore_index=True)
        meta_features_train_orig = meta_features_train_orig_par.append(meta_features_train_orig_3, ignore_index=True)
        
        ## Perform PCA for Training
        pca = PCA(n_components)
        pca.fit(meta_features_train_orig)
        meta_features_train_pca = pca.transform(meta_features_train_orig)
        #print(pca.explained_variance_ratio_)
        
        #print(meta_features_train_orig)
        #print(meta_features_train_pca)
        
        ## Normalize Features
        if PCA_option == 'No':
             meta_features_train = Normalize_feature(meta_features_train_orig, PCA_option)
        else:     
             meta_features_train = Normalize_feature(meta_features_train_pca, PCA_option)
 
        print('Normalized Train Features')
        print(meta_features_train)
        
        ## (1) Test Meta-features
        meta_features_test_orig_1 = Get_Meta_Features (meta_feat_dir_1, test_folds_list_uv[i], 'Uni-var')
        meta_features_test_orig_2 = Get_Meta_Features (meta_feat_dir_2, test_folds_list_uv[i], 'Uni-var')
        meta_features_test_orig_3 = Get_Meta_Features (meta_feat_dir_3, test_folds_list_uv[i], 'Uni-var')
        
        meta_features_test_orig_par = meta_features_test_orig_1.append(meta_features_test_orig_2, ignore_index=True)
        meta_features_test_orig = meta_features_test_orig_par.append(meta_features_test_orig_3, ignore_index=True)
        
        ## Perform PCA for Testing
        pca = PCA(n_components)
        pca.fit(meta_features_test_orig)
        meta_features_test_pca = pca.transform(meta_features_test_orig)
        #print(pca.explained_variance_ratio_)
        
        ## Normalize Features
        if PCA_option == 'No':
             meta_features_test = Normalize_feature(meta_features_test_orig, PCA_option)
        else:     
             meta_features_test = Normalize_feature(meta_features_test_pca, PCA_option)
 
        print('Normalized Test Features')    
        print(meta_features_test)
        
                       
    
        # (A) Define general meta-learning model
            
        #print(train_folds_list_uv[i])  
        
        start_time = time.time()    
        
        train_X = meta_features_train
        #train_Y_mse = []; train_Y_mape = []; train_Y_smape = []

        #df_meta_feat = df_meta_feat.fillna(0)
        
        ## Get the output vector for each dataset in that fold                 
        for train_data_name in train_folds_list_uv[i]: ## Append all of the first window performances      
            results_vec_mse_1, results_vec_mape_1, results_vec_smape_1, arr_models = Get_All_Model_Dataset(train_data_name, uv_dir_1, 'Uni-var')
            #train_Y_mse.append(results_vec_mse_1); train_Y_mape.append(results_vec_mape_1); train_Y_smape.append(results_vec_smape_1)
            
            ## Added For MV ranking results
            a_series = pd.Series(results_vec_mse_1)
            train_Y_mse = train_Y_mse.append(a_series, ignore_index=True)
            
            '''
            a_series = pd.Series(results_vec_mape_1)
            train_Y_mape = train_Y_mape.append(a_series, ignore_index=True)
            
            a_series = pd.Series(results_vec_smape_1)
            train_Y_smape = train_Y_smape.append(a_series, ignore_index=True)
            '''
            
        for train_data_name in train_folds_list_uv[i]: ## Append all of the second window performances    
            results_vec_mse_2, results_vec_mape_2, results_vec_smape_2, arr_models = Get_All_Model_Dataset(train_data_name, uv_dir_2, 'Uni-var')
            #train_Y_mse.append(results_vec_mse_2); train_Y_mape.append(results_vec_mape_2); train_Y_smape.append(results_vec_smape_2)
            
            a_series = pd.Series(results_vec_mse_2)
            train_Y_mse = train_Y_mse.append(a_series, ignore_index=True)
            
            '''
            a_series = pd.Series(results_vec_mape_2)
            train_Y_mape = train_Y_mape.append(a_series, ignore_index=True)
            
            a_series = pd.Series(results_vec_smape_2)
            train_Y_smape = train_Y_smape.append(a_series, ignore_index=True)
            '''
        
        for train_data_name in train_folds_list_uv[i]: ## Append all of the third window performances    
            results_vec_mse_3, results_vec_mape_3, results_vec_smape_3, arr_models = Get_All_Model_Dataset(train_data_name, uv_dir_3, 'Uni-var')
            #train_Y_mse.append(results_vec_mse_3); train_Y_mape.append(results_vec_mape_3); train_Y_smape.append(results_vec_smape_3)
            
            ## Added For MV ranking results
            a_series = pd.Series(results_vec_mse_3)
            train_Y_mse = train_Y_mse.append(a_series, ignore_index=True)
            
            '''
            a_series = pd.Series(results_vec_mape_3)
            train_Y_mape = train_Y_mape.append(a_series, ignore_index=True)
            
            a_series = pd.Series(results_vec_smape_3)
            train_Y_smape = train_Y_smape.append(a_series, ignore_index=True)
            '''
        
        ## Repeating the performance matrix to the multiple time windows
        #train_Y_mse = np.repeat(train_Y_mse, 3, axis=0)
        #train_Y_mape = np.repeat(train_Y_mape, 3, axis=0)
        #train_Y_smape = np.repeat(train_Y_smape, 3, axis=0)
        
        print(train_Y_mse)
        #break    
        #print (np.array(train_Y_mape).shape)
        
        
        # Fit Model for MSE Performance Metric 
        model_mse = LinearRegression(positive = True, normalize = True)
        #model_mse = Ridge(alpha = 1.0)
        
        
        train_X = train_X.fillna(0)
        train_Y_mse = train_Y_mse.fillna(0.01)
        
        model_mse.fit(train_X, train_Y_mse)
        print(model_mse.coef_)
        print(model_mse.score(train_X, train_Y_mse))
        
        
        AFGen_train_time_seconds = time.time() - start_time 
        print('AF-Gen Train Time: '+ str(AFGen_train_time_seconds))
        
        train_time_vec.append(AFGen_train_time_seconds) 
        
        print('Train_Time_Vector_After_Fold ' + str(i))
        print(train_time_vec)
        
        '''
        # Fit Model for MAPE Performance Metric 
        train_Y_mape = np.array(train_Y_mape)
        train_Y_mape[np.isinf(train_Y_mape)] = 100  ## Replacing infinity values with high number
        train_Y_mape[np.isnan(train_Y_mape)] = 100  ## Replacing NAN values with high number 

        model_mape = LinearRegression()
        model_mape.fit(train_X, train_Y_mape)
        print(model_mape.coef_)
        print(model_mape.score(train_X, train_Y_mape))
        
        # Fit Model for sMAPE Performance Metric 
        train_Y_smape = np.array(train_Y_smape)
        train_Y_smape[np.isinf(train_Y_smape)] = 100   ## Replacing infinity values with high number
        train_Y_smape[np.isnan(train_Y_smape)] = 100   ## Replacing NAN values with high number
        
        model_smape = LinearRegression(positive = True)
        model_smape.fit(train_X, train_Y_smape)
        print(model_smape.coef_)
        print(model_smape.score(train_X, train_Y_smape))
        '''
        
        '''
        start_time = time.time()
        
        # Make a Prediction
        test_X = meta_features_test
        yhat_mse = model_mse.predict(test_X)
        #yhat_mape = model_mape.predict(test_X) 
        #yhat_smape = model_smape.predict(test_X)
    
        
        # summarize prediction
        print(yhat_mse[0])
        print(yhat_mse)
        print(np.array(yhat_mse).shape)
        
        
        for test_idx in range(len(test_folds_list_uv[i])):
            
            test_Y_mse_1 = [];  test_Y_mse_2 = [];test_Y_mse_3 = []; ## Actual Performances (Ground Truth)
            #test_Y_mape_1 = []; test_Y_mape_2 = []; test_Y_mape_3 = []; 
            #test_Y_smape_1 = []; test_Y_smape_2 = []; test_Y_smape_3 = []; 
            
            
            ## Get the best model index from original data
            results_vec_mse_1, results_vec_mape_1, results_vec_smape_1, arr_models = Get_All_Model_Dataset(test_folds_list_uv[i][test_idx], uv_dir_1, 'Uni-var')   
            results_vec_mse_2, results_vec_mape_2, results_vec_smape_2, arr_models = Get_All_Model_Dataset(test_folds_list_uv[i][test_idx], uv_dir_2, 'Uni-var')  
            results_vec_mse_3, results_vec_mape_3, results_vec_smape_3, arr_models = Get_All_Model_Dataset(test_folds_list_uv[i][test_idx], uv_dir_3, 'Uni-var')   
            
            test_Y_mse_1.extend(results_vec_mse_1); test_Y_mse_2.extend(results_vec_mse_2); test_Y_mse_3.extend(results_vec_mse_3);
            #test_Y_mape_1.extend(results_vec_mape_1); test_Y_mape_2.extend(results_vec_mape_2); test_Y_mape_3.extend(results_vec_mape_3);
            #test_Y_smape_1.extend(results_vec_smape_1); test_Y_smape_2.extend(results_vec_smape_2); test_Y_smape_3.extend(results_vec_smape_3);
            
            ## Append three windows of MSE results
            a_series = pd.Series(test_Y_mse_1)
            test_Y_mse = test_Y_mse.append(a_series, ignore_index=True)
            
            a_series = pd.Series(test_Y_mse_2)
            test_Y_mse = test_Y_mse.append(a_series, ignore_index=True)
            
            a_series = pd.Series(test_Y_mse_3)
            test_Y_mse = test_Y_mse.append(a_series, ignore_index=True)
            
        '''
        '''
            ## Append three windows of MAPE results
            a_series = pd.Series(test_Y_mape_1)
            test_Y_mape = test_Y_mape.append(a_series, ignore_index=True)
            
            a_series = pd.Series(test_Y_mape_2)
            test_Y_mape = test_Y_mape.append(a_series, ignore_index=True)
            
            a_series = pd.Series(test_Y_mape_3)
            test_Y_mape = test_Y_mape.append(a_series, ignore_index=True)
            
            ## Append three windows of SMAPE results
            a_series = pd.Series(test_Y_smape_1)
            test_Y_smape = test_Y_smape.append(a_series, ignore_index=True)
            
            a_series = pd.Series(test_Y_smape_2)
            test_Y_smape = test_Y_smape.append(a_series, ignore_index=True)
            
            a_series = pd.Series(test_Y_smape_3)
            test_Y_smape = test_Y_smape.append(a_series, ignore_index=True)
        '''
        
        '''
        ## Get the best model index from Autoforecast general meta-learner 
            
            
        
        K = 5 ## Rank-K accuracy k
        a_vec = []
        cnt = 0
        for j in range(0, len(test_Y_mse)):
            res = sorted(range(len(test_Y_mse.iloc[j,:-1])), key = lambda sub: test_Y_mse.iloc[j,:-1][sub])[:K]
            #print(test_Y_mse.iloc[j,-1])
            print(res)
            print(np.argmin(yhat_mse[j]))
            if np.argmin(yhat_mse[j]) in res: #np.argmin(predicted_win[j]):
                cnt += 1
            #final_vec_all_folds.append(test_Y_mse.iloc[j,np.argmin(predicted_win[j])]) #predicted_mse.iloc[j,-1]])
            print(cnt)
        
        
        for idx in  range(3 * len(test_folds_list_uv[i])):
            ## Get the model index from the predicted general-regression model
            minimum_mse = np.argmin(yhat_mse[idx])
            #minimum_mape = np.argmin(yhat_mape[idx])
            #minimum_smape = np.argmin(yhat_smape[idx])
            #print(minimum_mse)
            
            ## Not needed for Inference time 
            results_vec_gen_mse.append(test_Y_mse.iloc[idx,minimum_mse])
            #results_vec_gen_mape.append(test_Y_mape.iloc[idx,minimum_mape])
            #results_vec_gen_smape.append(test_Y_smape.iloc[idx,minimum_smape])
            

       
        cnt_k_all += cnt
        print('Count-k: ' + str(cnt_k_all))
        print('Rank-k-Acc: ' + str((cnt_k_all / (len(test_Y_mse) * n_folds))))
        
        #print('MSE Results::::')
        #print(results_vec_gen_mse)
        #print(Average_val(results_vec_gen_mse))
    
        ## Estimate Inference Time
        AF_best_inference_run_time_seconds = time.time() - start_time 
        print('AutoForecast General Inference Time: '+ str(AF_best_inference_run_time_seconds))    
        '''
        
        '''
        print('MAPE Results::::')
        print(results_vec_gen_mape)
        print(Average_val(results_vec_gen_mape))
        
        print('SMAPE Results::::')
        print(results_vec_gen_smape)
        print(Average_val(results_vec_gen_smape))
        '''
        
        '''  
        ## Comparing indices
        predicted_mse = pd.DataFrame(yhat_mse)
            
        predicted_mse['MinColumnID']= predicted_mse.idxmin(axis=1)
        test_Y_mse['MinColumnID']= test_Y_mse.idxmin(axis=1)
        
        print(predicted_mse)
        print(test_Y_mse)
        
        print('Fold Finished ............')    
        '''
        
        #break
    

        



