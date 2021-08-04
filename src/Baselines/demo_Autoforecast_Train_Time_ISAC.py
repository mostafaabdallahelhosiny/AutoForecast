import pandas as pd
import os
import simplejson as json
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import time
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

def Average_val(lst):
    return sum(lst) / len(lst)

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
    
    #print(result_mse_0)
    return result_mse[0], result_mape[0], result_smape[0]
    
### Get Average Performance of Models across dataset cluster (Used for ISAC baseline)
def Get_Best_Avg_Model_Dataset(data_cluster, win_res_dir_name, data_type):
    
    results_vec_mse_avg = [0] * 322
    #results_vec_mape_avg = [0] * 322
    #results_vec_smape_avg = [0] * 322
    
    for data_name in data_cluster:
        
        results_vec_mse = []
        #results_vec_mape = []
        #results_vec_smape = []
        
        arr_models = os.listdir(win_res_dir_name + '/' + data_name)
        all_models_list = os.listdir(win_res_dir_name + '/' + data_name) 
    
        for model_name in all_models_list:
            df = pd.read_csv(win_res_dir_name + '/' + data_name+'/'+ model_name, error_bad_lines=False, header  = None)
            
            results_vec_mse.append(df.iloc[0,-1])
            #results_vec_mape.append(df.iloc[1,-1])
            #results_vec_smape.append(df.iloc[2,-1]) 
        
        results_vec_mse_avg  += results_vec_mse
        #results_vec_mape_avg  += results_vec_mape
        #results_vec_smape_avg  += results_vec_smape    
    
    best_model_dataset_index_mse_avg = results_vec_mse_avg.index(min(results_vec_mse_avg))
    #best_model_dataset_index_mape_avg = results_vec_mape_avg.index(min(results_vec_mape_avg))
    #best_model_dataset_index_smape_avg = results_vec_smape_avg.index(min(results_vec_smape_avg))
    
    
    return best_model_dataset_index_mse_avg #, best_model_dataset_index_mape_avg, best_model_dataset_index_smape_avg 

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

if __name__ == '__main__':
    
    n_folds = 5
    
    ## Folds for MV datasets
    mv_dir = 'results_all_mv/Multi-variate_first_win/'
    dir_mv_list = os.listdir(mv_dir)
    dir_mv_list.remove('.DS_Store')
    train_folds_list_mv, test_folds_list_mv = Divide_Dataset_Folds(n_folds, dir_mv_list)
    
    #print('MV<<<<') #print(train_folds_list_mv) #print(test_folds_list_mv)
     
    ## Folds for UV datasets
    #uv_dir_1 = 'results_all_uv/Uni-variate_first_win/' 
    #uv_dir_2 = 'results_all_uv/Uni-variate_second_win'
    #uv_dir_3 = 'results_all_uv/Uni-variate_third_win'
    
    
    
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
    
    #### Go across the folds and evaluate the different baselines and then average

    perf_mse_vec_global_best= []; perf_mape_vec_global_best = []; perf_smape_vec_global_best = []
    perf_mse_vec_algosmart = []; perf_mape_vec_algosmart = []; perf_smape_vec_algosmart = [] 
    perf_mse_vec_isac = []; perf_mape_vec_isac = []; perf_smape_vec_isac = []
    
    cnt_k_all = 0
    results_vec_gen_mse = []
    train_time_vec = []
    for i in range(0, n_folds): # n_folds
        #print(train_folds_list_uv[i])
        
        start_time = time.time()
        
        test_Y_mse = pd.DataFrame()
        train_Y_mse = pd.DataFrame()
            
                
        meta_features_train_orig_1 = Get_Meta_Features (meta_feat_dir_1, train_folds_list_uv[i], 'Uni-var')   ## Get Extracted Meta Features for 1st window
        meta_features_train_orig_2 = Get_Meta_Features (meta_feat_dir_2, train_folds_list_uv[i], 'Uni-var')   ## Get Extracted Meta Features for 2nd window
        meta_features_train_orig_3 = Get_Meta_Features (meta_feat_dir_3, train_folds_list_uv[i], 'Uni-var')   ## Get Extracted Meta Features for 3rd window
        
        meta_features_train_orig_par = meta_features_train_orig_1.append(meta_features_train_orig_2, ignore_index=True)
        meta_features_train = meta_features_train_orig_par.append(meta_features_train_orig_3, ignore_index=True)
        
        print(meta_features_train)
        
        
        meta_features_test_orig_1 = Get_Meta_Features (meta_feat_dir_1, test_folds_list_uv[i], 'Uni-var')
        meta_features_test_orig_2 = Get_Meta_Features (meta_feat_dir_2, test_folds_list_uv[i], 'Uni-var')
        meta_features_test_orig_3 = Get_Meta_Features (meta_feat_dir_3, test_folds_list_uv[i], 'Uni-var')
        
        meta_features_test_orig_par = meta_features_test_orig_1.append(meta_features_test_orig_2, ignore_index=True)
        meta_features_test = meta_features_test_orig_par.append(meta_features_test_orig_3, ignore_index=True)
   
        
        print(meta_features_test)
        
        ######### Baseline 0: No Model Selection
        # I have implemented that into another fine called "No_Model_Selection.py"
            
        
               
        ## Directory in which Multi-variate models results are stored
        mv_dir = 'results_all_mv/'
        dir_mv_list = os.listdir(mv_dir)
             
        ## Directory in which Uni-variate models results are stored
        uv_dir_all = 'results_all_uv/'
        dir_uv_list = os.listdir(uv_dir_all)
        data_type = 'Uni-var' 
        
        
        '''
        for test_idx in range(len(test_folds_list_uv[i])):
            
            test_Y_mse_1 = [];  test_Y_mse_2 = [];test_Y_mse_3 = []; ## Actual Performances (Ground Truth)
        
            
            ## Get the best model index from original data
            results_vec_mse_1, results_vec_mape_1, results_vec_smape_1, arr_models = Get_All_Model_Dataset(test_folds_list_uv[i][test_idx], uv_dir_1, 'Uni-var')   
            results_vec_mse_2, results_vec_mape_2, results_vec_smape_2, arr_models = Get_All_Model_Dataset(test_folds_list_uv[i][test_idx], uv_dir_2, 'Uni-var')  
            results_vec_mse_3, results_vec_mape_3, results_vec_smape_3, arr_models = Get_All_Model_Dataset(test_folds_list_uv[i][test_idx], uv_dir_3, 'Uni-var')   
            
            test_Y_mse_1.extend(results_vec_mse_1); test_Y_mse_2.extend(results_vec_mse_2); test_Y_mse_3.extend(results_vec_mse_3);
            
            ## Append three windows of MSE results
            a_series = pd.Series(test_Y_mse_1)
            test_Y_mse = test_Y_mse.append(a_series, ignore_index=True)
            
            a_series = pd.Series(test_Y_mse_2)
            test_Y_mse = test_Y_mse.append(a_series, ignore_index=True)
            
            a_series = pd.Series(test_Y_mse_3)
            test_Y_mse = test_Y_mse.append(a_series, ignore_index=True)
        '''
       
        ######### Baseline 2: ISAC
        
        print(len(train_folds_list_uv[i]))
        
        clustering = KMeans(n_clusters = 10)
        clustering.fit(meta_features_train)
        train_clusters = clustering.labels_
        predicted_clusters = clustering.predict(meta_features_test)
        
        ISAC_train_time_seconds = time.time() - start_time 
        print('ISAC Train Time: '+ str(ISAC_train_time_seconds))
        
        train_time_vec.append(ISAC_train_time_seconds) 
        
        print('Train_Time_Vector_After_Fold ' + str(i))
        print(train_time_vec)
        
        
        #print(predicted_clusters)
        #print(train_clusters)
        
        '''
        K = 1 ## Rank-K Accracy
        cnt = 0
        for j in range(len(test_folds_list_uv[i])):
            
                ## Get the closest cluster training datasets
                train_data_index = np.where(train_clusters==predicted_clusters[j])[0]
                
                #print(train_data_index)
                
                ## For mapping the time window to the dataset (recall, we have three windows for each dataset)
                for idx in range(0,len(train_data_index)):
                    if train_data_index[idx] < len(train_folds_list_uv[i]):
                        train_data_index[idx] = train_data_index[idx]
                    elif len(train_folds_list_uv[i]) <= train_data_index[idx] < int(2*len(train_folds_list_uv[i])):
                        train_data_index[idx] = train_data_index[idx] - len(train_folds_list_uv[i])
                    else:
                        train_data_index[idx] = train_data_index[idx] - int(2*len(train_folds_list_uv[i])) 
                          
                train_rel_cluster = []
                for train_idx in train_data_index:
                    train_rel_cluster.append(train_folds_list_uv[i][train_idx])
 
                print(train_rel_cluster)
                
                # Get the best average performing model on that cluster
                isac_model_index_mse = Get_Best_Avg_Model_Dataset(train_rel_cluster, uv_dir_1, 'Uni-var') #
                
                
                # Calculate the performance metric of the chosen model on the test dataset
                test_k1_data_name = test_folds_list_uv[i][j]
            
                res = sorted(range(len(test_Y_mse.iloc[j,:-1])), key = lambda sub: test_Y_mse.iloc[j,:-1][sub])[:K]
                print(res)
                if isac_model_index_mse in res: #np.argmin(predicted_win[j]):
                    cnt += 1

                print(cnt)
                
         
        print('ISAC Results Fold No. '+ str(i))  
        
        cnt_k_all += cnt
        print('Count-k' + str(cnt_k_all))
        print('Rank-k-Acc:' + str((cnt_k_all / (len(test_Y_mse) * n_folds))))
        '''
          
