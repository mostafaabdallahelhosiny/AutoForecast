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

    
    ## Multi-variate
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
    
    perf_mse_vec_global_best= []; perf_mape_vec_global_best = []; perf_smape_vec_global_best = []
    perf_mse_vec_algosmart = []; perf_mape_vec_algosmart = []; perf_smape_vec_algosmart = [] 
    perf_mse_vec_isac = []; perf_mape_vec_isac = []; perf_smape_vec_isac = []
    
    cnt_k_all = 0
    results_vec_gen_mse = []
    train_time_vec = []
    for i in range(0, n_folds): # n_folds
        #print(train_folds_list_uv[i])
        
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
        uv_dir_all = 'results_all_mv/' ##uv for uni-variate
        dir_uv_list = os.listdir(uv_dir_all)
        data_type = 'Uni-var' 
        
        
        ######### Baseline 1: Global Best
        ## Start Time of Training the model
        start_time = time.time()
        
        ## Baseline: SS
        
        train_X = meta_features_train
        #train_Y_mse = []; train_Y_mape = []; train_Y_smape = []

        #df_meta_feat = df_meta_feat.fillna(0)
        
        ## Get the output vector for each dataset in that fold                 
        for train_data_name in train_folds_list_uv[i]: ## Append all of the first window performances      
            results_vec_mse_1, results_vec_mape_1, results_vec_smape_1, arr_models = Get_All_Model_Dataset(train_data_name, uv_dir_1, 'Uni-var')
            #train_Y_mse.append(results_vec_mse_1); 
            
            a_series = pd.Series(results_vec_mse_1)
            train_Y_mse = train_Y_mse.append(a_series, ignore_index=True)
            
            
        for train_data_name in train_folds_list_uv[i]: ## Append all of the second window performances    
            results_vec_mse_2, results_vec_mape_2, results_vec_smape_2, arr_models = Get_All_Model_Dataset(train_data_name, uv_dir_2, 'Uni-var')
            #train_Y_mse.append(results_vec_mse_2); 
            
            
            a_series = pd.Series(results_vec_mse_2)
            train_Y_mse = train_Y_mse.append(a_series, ignore_index=True)
            
        
        for train_data_name in train_folds_list_uv[i]: ## Append all of the third window performances    
            results_vec_mse_3, results_vec_mape_3, results_vec_smape_3, arr_models = Get_All_Model_Dataset(train_data_name, uv_dir_3, 'Uni-var')
            #train_Y_mse.append(results_vec_mse_3);
            
            
            a_series = pd.Series(results_vec_mse_3)
            train_Y_mse = train_Y_mse.append(a_series, ignore_index=True)
        
        print(train_Y_mse)
        
        
        train_X = train_X.fillna(0)
        train_Y_mse = train_Y_mse.fillna(0)
        
        clf = MLPRegressor(random_state=1, max_iter=500)                           
        clf.fit(train_X, train_Y_mse)
        
        
        #test_pred = clf.predict(X_test).reshape(1,-1)
        
        SS_train_time_seconds = time.time() - start_time 
        print('SS Train Time: '+ str(SS_train_time_seconds))
        
        train_time_vec.append(SS_train_time_seconds) 
        
        print('Train_Time_Vector_After_Fold ' + str(i))
        print(train_time_vec)
        '''
        ## Commented Inference for Now
        start_time = time.time()
        
        # Make a Prediction
        test_X = meta_features_test
        yhat_mse = clf.predict(test_X)
        #test_pred[test_pred < 0] = 0
        
        # summarize prediction
        print(yhat_mse[0])
        print(yhat_mse)
        print(np.array(yhat_mse).shape)
        
        
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
            
        
        K = 50
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
       
            ## Not needed for Inference time 
            results_vec_gen_mse.append(test_Y_mse.iloc[idx,minimum_mse])
           
        
       
        cnt_k_all += cnt
        print('Count-k: ' + str(cnt_k_all))
        print('Rank-k-Acc: ' + str((cnt_k_all / (len(test_Y_mse) * n_folds))))
        
        print('MSE Results::::')
        #print(results_vec_gen_mse)
        #print(Average_val(results_vec_gen_mse))
        
        
        ## Estimate Inference Time
        SS_best_inference_run_time_seconds = time.time() - start_time 
        print('SS Inference Time: '+ str(SS_best_inference_run_time_seconds))
        '''
    
        
        '''        
        glob_best_model_name_mse, glob_best_model_name_mape, glob_best_model_name_smape = Global_Best(uv_dir_all, train_folds_list_uv[i], data_type)
        #print(glob_best_model_name_mape)
        #print(glob_best_model_name_smape)
        #print(glob_best_model_name_mse)
        
        
        for j in range(len(test_folds_list_uv[i])):
            
                # Calculate the performance metric of the chosen global best model on the test dataset
                test_k1_data_name = test_folds_list_uv[i][j]
            
                df_test_mse_global_best = pd.read_csv(uv_dir +  test_k1_data_name + '/'+ glob_best_model_name_mse, error_bad_lines=False, header  = None)
                perf_mse_vec_global_best.append(df_test_mse_global_best.iloc[0,-1]) ## MSE 
            
                df_test_mape_global_best = pd.read_csv(uv_dir + test_k1_data_name + '/'+ glob_best_model_name_mape, error_bad_lines=False, header  = None)
                perf_mape_vec_global_best.append(df_test_mape_global_best.iloc[1,-1]) ## MAPE
            
                df_test_smape_global_best = pd.read_csv(uv_dir + test_k1_data_name + '/'+ glob_best_model_name_smape, error_bad_lines=False, header  = None)
                perf_smape_vec_global_best.append(df_test_smape_global_best.iloc[2,-1]) ## SMAPE
        
        print('GloBAL Best Results Fold No. '+ str(i))
        print(perf_mse_vec_global_best)
        print(perf_mape_vec_global_best)
        print(perf_smape_vec_global_best)
    
   

    ## Calculating Run time of the model inference process
    global_best_inference_run_time_seconds = time.time() - start_time    
    print('GB Inference Time: '+ str(global_best_inference_run_time_seconds)) 
    ''' 
    '''   
        ######### Baseline 2: ISAC
        print(len(train_folds_list_uv[i]))
        
        clustering = KMeans(n_clusters = 10)
        clustering.fit(meta_features_train)
        train_clusters = clustering.labels_
        predicted_clusters = clustering.predict(meta_features_test)
        
        print(predicted_clusters)
        print(train_clusters)
        
        for j in range(len(test_folds_list_uv[i])):
            
                ## Get the closest cluster datasets
                train_data_index = np.where(train_clusters==predicted_clusters[j])[0]
                
                print(train_data_index)
                
                ## For mapping the time window to the dataset (recall, we have three windows for each dataset)
                for idx in range(0,len(train_data_index)):
                    if train_data_index[idx] < 257:
                        train_data_index[idx] = train_data_index[idx]
                    elif 257 <= train_data_index[idx] < int(2*257):
                        train_data_index[idx] = train_data_index[idx] - 257
                    else:
                        train_data_index[idx] = train_data_index[idx] - int(2*257) 
                          
                print(train_data_index)       
                print (np.array(train_folds_list_uv[i])[train_data_index])
                
                train_rel_cluster = []
                for train_idx in train_data_index:
                    train_rel_cluster.append(train_folds_list_uv[i][train_idx])
 
                print(train_rel_cluster)
                
                # Get the best average performing model on that cluster
                isac_model_name_mse, isac_model_name_mape, isac_model_name_smape = Get_Best_Avg_Model_Dataset(train_rel_cluster, uv_dir, 'Uni-var')
                
                
                # Calculate the performance metric of the chosen model on the test dataset
                test_k1_data_name = test_folds_list_uv[i][j]
            
                df_test_mse_isac = pd.read_csv(uv_dir + '/' + test_k1_data_name + '/'+ isac_model_name_mse, error_bad_lines=False, header  = None)
                perf_mse_vec_isac.append(df_test_mse_isac.iloc[0,-1]) ## MSE 
            
                df_test_mape_isac = pd.read_csv(uv_dir + '/' + test_k1_data_name + '/'+ isac_model_name_mape, error_bad_lines=False, header  = None)
                perf_mape_vec_isac.append(df_test_mape_isac.iloc[1,-1]) ## MAPE
            
                df_test_smape_isac = pd.read_csv(uv_dir + '/' + test_k1_data_name + '/'+ isac_model_name_smape, error_bad_lines=False, header  = None)
                perf_smape_vec_isac.append(df_test_smape_isac.iloc[2,-1]) ## SMAPE
            
         
        print('ISAC Results Fold No. '+ str(i))    
        print(perf_mse_vec_isac)
        print(perf_mape_vec_isac)
        print(perf_smape_vec_isac)
    
    ISAC_best_inference_run_time_seconds = time.time() - start_time    
    print('ISAC Inference Time: '+ str(ISAC_best_inference_run_time_seconds))                    
        
    '''
        
    '''
        ###### Baseline 3: ALGOSMART
        neigh = NearestNeighbors(n_neighbors=5)
        neigh.fit(meta_features_train) ## May need PCA
    
        neighbors = neigh.kneighbors(meta_features_test, 5, return_distance=False)
        print(neighbors)
        
        # only select the k==1 case one
        k1_neighbors = neighbors[:, 0]
        print(k1_neighbors)
        
        ## Mapping the windows to the datasets (since each dataset has three windows)
        for idx in range(0,len(k1_neighbors)):
            if k1_neighbors[idx] < 257:
                k1_neighbors[idx] = k1_neighbors[idx]
            elif 257 <= k1_neighbors[idx] < int(2*257):
                k1_neighbors[idx] = k1_neighbors[idx] - 257
            else:
                k1_neighbors[idx] = k1_neighbors[idx] - int(2*257) 
        
        print(k1_neighbors)
        
        test_data_idx = 0
        for train_k1 in k1_neighbors:
            
            print(train_k1)
            # Get the 1NN dataset Name
            train_k1_data_name = train_folds_list_uv[i][train_k1]
             
            # Get the corresponding best model from 1NN dataset
            model_name_mse, model_name_mape, model_name_smape = Get_Best_Model_Dataset(train_k1_data_name, uv_dir, 'Uni-var')
            print(model_name_mse)
            
            
            # Calculate the performance metric of the chosen model on the test dataset
            print(test_data_idx)
            test_k1_data_name = test_folds_list_uv[i][test_data_idx]
            
            df_test_mse = pd.read_csv(uv_dir + '/' + test_k1_data_name + '/'+ model_name_mse, error_bad_lines=False, header  = None)
            perf_mse_vec_algosmart.append(df_test_mse.iloc[0,-1]) ## MSE  _algosmart
            
            df_test_mape = pd.read_csv(uv_dir + '/' + test_k1_data_name + '/'+ model_name_mape, error_bad_lines=False, header  = None)
            perf_mape_vec_algosmart.append(df_test_mape.iloc[1,-1]) ## MAPE
            
            df_test_smape = pd.read_csv(uv_dir + '/' + test_k1_data_name + '/'+ model_name_smape, error_bad_lines=False, header  = None)
            perf_smape_vec_algosmart.append(df_test_smape.iloc[2,-1]) ## SMAPE
            
            test_data_idx += 1
            
            if test_data_idx  < 64:
                test_data_idx = test_data_idx
            elif 64 <= test_data_idx < int(2*64):
                test_data_idx -= 64
            else:
                test_data_idx -=  int(2 * 64)
            
       
        print('ALGOSMART Results Fold No.'+ str(i))  
        
        print(perf_mse_vec_algosmart)
        print(perf_mape_vec_algosmart)
        print(perf_smape_vec_algosmart)
    
    
    AS_best_inference_run_time_seconds = time.time() - start_time    
    print('AS Inference Time: '+ str(AS_best_inference_run_time_seconds))
    
    
    print('MSE::')    
    print(Average_val(perf_mse_vec_algosmart))
    
    print('MAPE::')    
    print(Average_val(perf_mape_vec_algosmart))
    
    print('sMAPE::')    
    print(Average_val(perf_smape_vec_algosmart))
    '''
        
        
        
    
#################################################################
    ### Now, checking the different baselines

    ## (a) Global_Best
    a_all_mv = [296, 251, 49, 3, 40, 190, 179, 162, 99, 191, 317, 92, 65, 168, 84, 205, 173, 51, 136, 192, 63, 24, 166, 296, 200, 162, 39, 39, 268, 100, 179, 63, 39, 275, 278, 191, 291, 138, 19, 191, 40, 120, 63, 191, 61, 179, 275, 275, 191, 168, 1, 21, 278, 5, 68, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 35, 156, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 192, 40, 5, 171, 61, 51, 191, 51, 152, 247, 39, 225, 4, 16, 60, 318, 5, 152, 315, 232, 21, 315, 49, 21, 275, 63, 272, 84, 315, 315, 39, 251, 84, 316, 278, 61, 291, 99, 1, 200, 308, 225, 3, 90, 142, 183, 171, 308, 1, 138, 263, 92, 96, 61, 179, 39, 205, 16, 43, 58, 61, 206, 166, 17, 152, 90, 317, 110, 21, 63, 39, 21, 168, 179, 138, 71, 179, 291, 278, 100, 39, 247, 40, 311, 306, 39, 68, 82, 20, 200, 84, 39, 99, 110, 34, 275, 317, 100, 191, 57, 37, 49, 49, 212, 308, 114, 183, 278, 13, 55, 182, 82, 272, 125, 1, 311, 114, 225, 63, 183, 1, 21, 143, 315, 212, 152, 315, 225, 65, 1, 291, 259, 17, 68, 272, 173, 56, 100, 1, 81, 1, 1, 251, 241, 39, 39, 60, 20, 183, 241, 278, 39, 13, 1, 4, 34, 171, 200, 285, 315, 308, 251, 126, 55, 192, 42, 42, 1, 39, 278, 13, 64, 81, 315, 120, 236, 315, 60, 58, 120, 268, 57, 58, 275, 181, 113, 13, 69, 19, 179, 8, 246, 21, 152, 163, 75, 38, 51, 42, 315, 263, 35, 110, 65, 65, 312, 148, 60, 251, 61, 49, 49, 16, 13, 1, 1, 13, 13, 13, 2, 209, 13, 1, 181, 296, 69, 49, 42, 265, 34, 110, 5, 69, 278, 81, 306, 45, 265, 1, 34, 200, 156, 60, 1, 55, 272, 69, 1, 65, 49, 171, 183, 9, 15, 32, 63, 272, 1, 278, 247, 38, 5, 36, 306, 182, 247, 100, 182, 1, 306, 5, 272, 20, 68, 5, 68, 39, 152, 268, 9, 32, 14, 9, 179, 322, 306, 163, 251, 60, 101, 82, 32, 100, 316, 247, 225, 1, 21, 173, 39, 229, 278, 168, 229, 39, 251, 58, 254, 272, 110, 263, 82, 311, 190, 19, 296, 38, 179, 272, 205, 39, 21, 39, 241, 263, 101, 90, 5, 236, 40, 246, 60, 40, 42, 316, 138, 1, 278, 171, 156, 55, 84, 21, 265, 278, 256, 1, 110, 34, 5, 49, 45, 42, 163, 306, 39, 173, 2, 96, 39, 1, 110, 1, 181, 241, 60, 39, 34, 1, 60, 34, 263, 5, 200, 101, 19, 183, 322, 2, 179, 181, 17, 16, 120, 110, 152, 51, 190, 244, 17, 5, 316, 162, 120, 1, 272, 168, 1, 17, 143, 45, 143, 5, 42, 21, 42, 57, 58, 110, 34, 32, 32, 21, 254, 49, 225, 110, 60, 63, 32, 183, 81, 63, 21, 21, 268, 268, 171, 278, 4, 82, 212, 278, 247, 110, 1, 1, 32, 143, 254, 69, 109, 156, 1]
    a_all_uv_win1 = [171, 54, 246, 73, 191, 152, 179, 251, 138, 136, 315, 182, 281, 13, 38, 72, 1, 272, 315, 39, 101, 8, 4, 21, 247, 13, 54, 40, 1, 13, 35, 275, 278, 315, 19, 2, 315, 4, 242, 171, 60, 100, 296, 236, 63, 92, 100, 40, 291, 16, 138, 45, 306, 120, 58, 263, 1, 120, 13, 136, 243, 312, 64, 13, 84, 61, 82, 35, 68, 39, 136, 163, 315, 120, 63, 138, 5, 68, 102, 296, 17, 306, 100, 72, 166, 13, 251, 1, 311, 4, 291, 247, 64, 168, 163, 236, 38, 15, 311, 152, 183, 179, 212, 54, 183, 49, 40, 268, 195, 278, 183, 182, 263, 143, 280, 316, 156, 278, 21, 263, 81, 311, 24, 152, 263, 163, 291, 99, 63, 315, 63, 311, 317, 82, 19, 13, 4, 55, 19, 315, 179, 225, 125, 125, 4, 1, 36, 36, 84, 256, 13, 68, 82, 13, 95, 308, 171, 19, 236, 2, 8, 1, 291, 163, 254, 171, 311, 247, 315, 254, 275, 171, 84, 66, 68, 101, 100, 73, 114, 19, 44, 55, 308, 317, 263, 272, 69, 64, 38, 101, 1, 24, 263, 54, 39, 242, 317, 183, 247, 136, 138, 83, 51, 13, 115, 142, 19, 39, 275, 9, 311, 272, 13, 312, 163, 168, 163, 291, 236, 57, 251, 306, 13, 39, 308, 315, 171, 278, 73, 312, 21, 236, 82, 74, 190, 36, 278, 315, 236, 19, 247, 38, 39, 136, 68, 162, 84, 225, 315, 236, 318, 100, 99, 81, 181, 171, 205, 60, 138, 241, 1, 232, 317, 191, 15, 39, 39, 54, 268, 184, 278, 69, 296, 247, 40, 110, 1, 56, 162, 13, 191, 60, 315, 16, 254, 251, 110, 16, 143, 60, 100, 120, 272, 182, 168, 16, 49, 73, 21, 152, 69, 120, 316, 190, 200, 268, 42, 110, 162, 317, 1, 60, 3, 110, 148, 236, 190, 58, 254, 13, 40, 83, 110, 82, 68]
    a_all_mv.extend(a_all_uv_win1)

#    print(Global_Best (a_all_mv))
    
    print(len(a_all_mv))
    Histogram_plot_save(a_all_mv)


    ## (b) 






