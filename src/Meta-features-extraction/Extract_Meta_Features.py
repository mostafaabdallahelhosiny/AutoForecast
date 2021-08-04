import pandas as pd
import os
import simplejson as json
import csv
import numpy as np
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from sklearn import preprocessing
from gluonts.dataset.util import to_pandas
import pprint
import re
from scipy.stats import skew, kurtosis
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.mx.trainer import Trainer
import time
from gluonts.model.prophet import ProphetPredictor 
#from prophet import Prophet
from gluonts.model.seasonal_naive import SeasonalNaivePredictor        
from gluonts.model.deep_factor import DeepFactorEstimator
from gluonts.model.n_beats import NBEATSEstimator
from gluonts.model.gp_forecaster import GaussianProcessEstimator
# VAR example
from statsmodels.tsa.vector_ar.var_model import VAR
from random import random
# AR example
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tools.eval_measures import mse as mean_square_error

from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

from tsfresh import extract_relevant_features
from sklearn.decomposition import PCA

from pandas import DataFrame
from pandas import concat
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import *
from sklearn.linear_model import BayesianRidge



def list_process_name(var):
    return [var+'_min', var+'_max', var+'_mean', var+'_std', var+'_skewness', var+'_kurtosis']

def list_process(x, r_min=True, r_max=True, r_mean=True, r_std=True, r_skew=True, r_kurtosis=True):
    x = np.asarray(x).reshape(-1, 1)
    return_list = []
    
    if r_min:
        return_list.append(np.nanmin(x))

    if r_max:
        return_list.append(np.nanmax(x))

    if r_mean:
        return_list.append(np.nanmean(x))
        
    if r_std:
        return_list.append(np.nanstd(x))
        
    if r_skew:
        return_list.append(skew(x, nan_policy='omit')[0])
        
    if r_kurtosis:
        return_list.append(kurtosis(x, nan_policy='omit')[0])
        
    return return_list


def symm_mean_absolute_percentage_error(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)

def mean_absolute_percentage_error(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100


def Generate_Inference_Metrics(predictor, test_data, Num_Sample_Paths):
    ## Start Time of inference using the model
    start_time = time.time()
            
    ## Forecasting using the built forecasting model
    forecast_it, ts_it = make_evaluation_predictions(
            dataset= test_data ,  # test dataset
            predictor= predictor,  # predictor
            num_samples= Num_Sample_Paths,  # number of sample paths we want for evaluation
            )

    ## Convert the forecast into list for better processing 
    forecasts = list(forecast_it)
    tss = list(ts_it)
            
    ## Calculating Run time of the inference process
    inference_run_time_seconds = time.time() - start_time
    
    
    ## Computing the forecasting metrics (Aggregate metrics aggregate both across time-steps and across time series)
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_data))
    smape = agg_metrics['sMAPE']
    mape  = agg_metrics['MAPE']
    mse = agg_metrics['MSE']
            
    return inference_run_time_seconds, smape, mape, mse 
            
# transform a time series dataset into a supervised learning dataset for forecasting
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    #n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    #print(agg)
    return agg.values


def get_files_by_file_size(dirname, reverse=False):
    """ Return list of file paths in directory sorted by file size """

    # Get list of files
    filepaths = []
    for basename in os.listdir(dirname):
        filename = os.path.join(dirname, basename)
        if os.path.isfile(filename):
            filepaths.append(filename)

    # Re-populate list with filename, size tuples
    for i in range(len(filepaths)):
        filepaths[i] = (filepaths[i], os.path.getsize(filepaths[i]))

    # Sort list by file size
    # If reverse=True sort from largest to smallest
    # If reverse=False sort from smallest to largest
    filepaths.sort(key=lambda filename: filename[1], reverse=reverse)

    # Re-populate list with just filenames
    for i in range(len(filepaths)):
        filepaths[i] = filepaths[i][0]

    return filepaths


if __name__ == '__main__':  
    
    ### Main Variables
    
    lag_par = 16
    Num_points = lag_par + 1 ## Controls Number of points to extract meta-features
    start_ind = int(lag_par - 12)
    end_ind =   int(lag_par + 3) + 1 # start_ind = int(lag[0] + 1) 

    print(start_ind)
    print(end_ind)

    

    ## Read the datasets directory
    dir_datasets_name = 'data-types-final/Uni_Variate/'
    
    #dir_datasets = os.listdir(dir_datasets_name)
    
    dir_datasets_sorted = get_files_by_file_size(dir_datasets_name, reverse=False) ## Sort files by file size from smallest to largest

    cnt = 0 ## Count of evaluated datasets
    
    for file_name in dir_datasets_sorted:
            meta_vec = []
            meta_vec_names = []
            
            ## Skip '.DS_Store' files of Mac OS
            if '.DS_Store' in file_name:
                    continue
            
            print(file_name)
    
        ## Loading the CSV dataset
        #cnt += 1
        #try:
            df = pd.read_csv(file_name, error_bad_lines=False, header  = None)
            
        
            ## Filling NaN values with 0
            df = df.fillna(0)
        
            ## Remove any lines with alphabet texts or special characters 
            df = df[pd.to_numeric(df[0], errors='coerce').notnull()]
            
            df = df.iloc[start_ind: end_ind + 1,:]
            
            #print(len(df))

            ## Normalizatio of the dataframe
            x = df.values 
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            df = pd.DataFrame(x_scaled)
            
            df_new = df
         
            if len(df) < lag_par:
                continue
            
            cnt += 1
            
    
            ## Start Time of Training the model
            start_time = time.time()
 
            
            #print(df.head())
            df['values'] = df[0]
            df["type"] = 1 ## All rows same class
            df.reset_index(inplace=True)
            
            
            ## Extract Time-series Meta-Features
            extracted_features = extract_features(df, column_id = 'type',column_sort='index',column_value = 'values')
            
            impute(extracted_features) ## Put zeros on all features with NaN values
            
            
            meta_vec.extend(extracted_features.iloc[0,:].tolist())
            meta_vec_names.extend(extracted_features.columns)
            
            #print(meta_vec)
            #print(meta_vec_names)
            
            ##### Extract Landmarker Features
            
            ## (A) Random Forest Landmarker Features
            
            ## Train Data Preparation
            print(df_new)
            train_data = df_new.iloc[0: Num_points + 1, 1] ## 2 * 
            print(train_data)
            # transform the time series data into supervised learning
            train = series_to_supervised(train_data, n_in = lag_par)
            
            # split into input and output columns
            trainX, trainy = train[:, :-1], train[:, -1]
            
            
            n_estimators = 100
            model = RandomForestRegressor(n_estimators =n_estimators, oob_score = True) #n_estimators= i
            
            model.fit(trainX, trainy)
            
            #### Extract Landmarker Features from RandomForest
            n_leaves = []
            n_depth = []
            fi_mean = []
            fi_max = []
    
            # doing this for each sub-trees 
            for i in range(n_estimators):
                n_leaves.append(model.estimators_[i].get_n_leaves())
                n_depth.append(model.estimators_[i].get_depth())
                fi_mean.append(model.estimators_[i].feature_importances_.mean())
                fi_max.append(model.estimators_[i].feature_importances_.max())
                # print(clf.estimators_[i].tree_)
            
            meta_vec.extend(list_process(n_leaves))
            meta_vec.extend(list_process(n_depth))
            meta_vec.extend(list_process(fi_mean))
            meta_vec.extend(list_process(fi_max))
            meta_vec.append(model.oob_score_)
            
            
            meta_vec_names.extend(list_process_name('RForest_n_leaves'))
            meta_vec_names.extend(list_process_name('RForest_n_depth'))
            meta_vec_names.extend(list_process_name('RForest_fi_mean'))
            meta_vec_names.extend(list_process_name('RForest_fi_max'))
            meta_vec_names.append('RForest_oob_score')
            #print(model.oob_prediction_)
            #print(model.get_params(deep = True))
            
            
            ### Extract Landmarker Features from Bayesian Ridge Regression
            
            clf = BayesianRidge(compute_score=True)
            clf.fit(trainX, trainy)
            
            meta_vec.append(clf.coef_[-1])
            meta_vec.append(clf.scores_[-1])
            meta_vec.append(clf.lambda_)    
            meta_vec.append(clf.alpha_)
            meta_vec.append(clf.n_iter_)
       
            meta_vec_names.append('BRidge_coef')
            meta_vec_names.append('BRidge_log_marginal_likelihood')
            meta_vec_names.append('BRidge_weight_precision')
            meta_vec_names.append('BRidge_noise_precision')
            meta_vec_names.append('BRidge_number_iter')      

            #print(meta_vec) #print(meta_vec_names) #print(len(meta_vec_names)) #print(len(meta_vec))
            
            numpy_meta_vec = np.array(meta_vec).reshape((1, 817))
            numpy_meta_vec_names = np.array(meta_vec_names).reshape((1, 817))
        
            ## Writing dataframe that contains all meta-features to CSV File
            final_meta_df = DataFrame(numpy_meta_vec, columns = numpy_meta_vec_names.T)

            final_meta_df.to_csv('Meta-Features/Meta-Feat_sec_win/Uni-Variate_sec_win/' + file_name[file_name.rfind('/')+1:file_name.index('.csv')]+'.csv', index=False)
                        
            ## Calculating Run time of the model training process
            run_time_seconds = time.time() - start_time
            print('Extracting Features Time: ' + str(run_time_seconds))
            
            
            
    #        if cnt >= 2:
    #            break
            print(cnt)
    
    #    except: ## if the file is corrupt or empty skip dataset
    #        continue
        

###########################################################################################################################################
                    
            ''''
            ## Select relevznt features and remove the rest with NAN values (due to low statistics)
            impute(extracted_features)
            features_filtered = select_features(extracted_features,y)
                
            ## Calculating Run time of the model training process
            run_time_seconds = time.time() - start_time
            print('Extracting Features Time: ' + str(run_time_seconds))
            
            print(featues_filtered)            
          
            '''
     