import pandas as pd
import datetime
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
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.mx.trainer import Trainer
import time
from gluonts.model.prophet import ProphetPredictor 
from fbprophet import Prophet
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
import itertools

from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import SimpleExpSmoothing,ExponentialSmoothing
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt



def Expontial_Smoothing(data, smooth_level):
    
    # Fit Exponential Smoothing 
    model = ExponentialSmoothing(data).fit(smoothing_level= smooth_level,optimized=False)
    
    # Return Fitted (smoothed) Values
    data_smoothed = model.fittedvalues
    
    return data_smoothed


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
    print(agg)
    return agg.values

   
def Write_Performance_File(dataset_name, model_name, lag, mse, mape, smape, train_time, infer_time):
    
    mse_vec  = ['MSE']
    mape_vec = ['MAPE']
    smape_vec = ['SMAPE']
    train_time_vec = ['Training Time']
    infer_time_vec = ['Inference Time']
    
    mse_vec.append(mse)
    mape_vec.append(mape)
    smape_vec.append(smape)
    train_time_vec.append(train_time)
    infer_time_vec.append(infer_time)
    
    ## Writing Performance Matrices into CSV Files
    dataset_name_no_dir = dataset_name[dataset_name.rfind('/')+1:dataset_name.index('.csv')]
    
    with open('results/'+ dataset_name_no_dir +'/' + model_name + '_lag=' + str(lag)+ '.csv', 'w', ) as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)                
        wr.writerows([mse_vec])
        wr.writerows([mape_vec])
        wr.writerows([smape_vec])
        wr.writerows([train_time_vec])
        wr.writerows([infer_time_vec])
        
        
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
            
'''            
def writing_performance_matrix_headers(model_name, param_list_1,param_list_2):
    
    ## Full Matrix with Models 
    full_param = []
    full_param_headers = []
    
    config_headers = 300    

    param_mat = np.zeros([config_headers, 14]).astype('object')
    param_tracker = 0

    param_list = []

    for r in itertools.product(param_list_1, param_list_2): 
        param_list.append((r[0], r[1]))
        param_mat[param_tracker, 0] = r[0]
        param_mat[param_tracker, 1] = r[1]
        param_tracker+=1

    full_param.extend(param_list)
    full_param_headers.extend([model_name]*len(param_list))
    
    print(full_param)
    print(full_param_headers)
'''    
                


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
    
    ## Name of Performance Matrix File    
    mse_vec  = ['MSE']
    mape_vec = ['MAPE']
    smape_vec = ['SMAPE']
    train_time_vec = ['Training Time']
    infer_time_vec = ['Inference Time']
    

    ### (A) Define the Models and Hyper-parameters Dictionaries (Model_Name: Hyper_parameter)
    ## Common Parameters Across Models
    Num_Sample_Paths = 100          ## A sample path is a potential realization of the time series into the future
    lag = [16]               ## Lag Parameter (4,8,32)
    start_ind = int(lag[0] - 12) ## int(2 * lag[0]) - 5 #
    print(start_ind)
    end_ind  = int(lag[0] + 3) #+ 10 
    print(end_ind)
    prediction_length = 1           ## Length of Prediction. Known also as "Forecasting Horizon"
    freq = "5min"                   ## Data Frequency (Kept Constant for all models)
    preprocess = ["No", "Exp_Smoothing"] #"Exp_Smoothing" ## Data Representation or Pre-Processing
    smooth_level = 0.4 ## Exponential Smoothing Value.: Alpha is often set to a value between 0 and 1. Large values mean that the model pays attention mainly to the most recent past observations, whereas smaller values mean more of the history is taken into account when making a prediction.
    
    
    ## (1) DeepAR Hyper-Parameters 
    model_name_DeepAR = "DeepAR" 
    DEEPAR_epochs = 5              ## That is number of training epochs
    DEEPAR_param_list_1 = [10,20,30,40,50]      ## DEEPAR num_cells
    DEEPAR_param_list_2 = [1,2,3,4,5]           # DEEPAR num_rnn_layers
    
    ## (2) Prophet Hyper-Parameters
    model_name_Prophet = "Prophet"
    #Prophet_param_list_1 = ["H","D","M","1min"] ## Prophet_freq = "5min"
    Prophet_param_list_1 = [0.001, 0.01, 0.1, 0.2, 0.5]  #'changepoint_prior_scale': determines flexibility of trend, and how much trend changes at trend changepoints.
    Prophet_param_list_2 = [0.01, 0.1, 1.0, 5.0, 10.0]        #'seasonality_prior_scale': This parameter controls the flexibility of the seasonality.
     
    
    ## (3) Seasonal Naive Hyper-parameterss
    model_name_Seasonal_Naive = "Seasonal Naive" 
    Seasonal_Naive_param_list_1 = [1,5,7,10,30] ## season length
     
    
    ## (4) DeepFactor Hyper-parameters
    model_name_DF = "DeepFactor" 
    DF_param_list_1 = [10,20,30,40,50]       # DF_num_hidden_global: Number of units per hidden layer for the global RNN model (default: 50).
    DF_param_list_2 = [1,5, 10, 15,20]           # DF_num_global_factors: Number of global factors (default: 10).
  
    ## (5) Gaussian Process Hyper-parameters
    model_name_GP = "Gaussian_Process" 
    GP_param_list_1 = [2,4,6,8,10]          # GP_cardinality Number of time series.
    GP_param_list_2 = [5,10,15,20,25]          # GP_max_iter_jitter = [5,10]
    
    
    ## (6) Auto Regression
    model_name_ar = "AR"
    AR_param_list_1 = ['HC0','HC1', 'HC2', 'HC3', 'nonrobust']      ## AR_cov_type: Maximum Likelihood or other types
    AR_param_list_2 = ['n', 'c', 't', 'ct'] #The trend to include in the model:
    
    
    ## (7) Random Forest Regressor
    RF_param_list_1 = [10,50,100,250,500,1000]  ## Number of Trees in the Forest
    RF_param_list_2 = [2,5,10,25,50, 'None']  ## The maximum depth of the tree


    ## (B) Read the datasets directory and Load Datasets
    dir_datasets_name = 'data-types-final/Uni_Variate'#'data-cleaned/MHSETS' #'data-cleaned/TCPD_datasets/'  #'data-cleaned/MHMSETS' #'data-cleaned/Adobe_Service_CPU_Memory_datasets_7days' #data-cleaned/Mustafa Prev time-series datasets
    #dir_datasets = os.listdir(dir_datasets_name)
    
    
    dir_datasets_sorted = get_files_by_file_size(dir_datasets_name, reverse=False) ## Sort files by file size from smallest to largest
    
    print(dir_datasets_sorted)

    cnt = 0 ## Count of evaluated datasets
    
    for file_name in dir_datasets_sorted:
        
        
        ## Skip '.DS_Store' files of Mac OS
        if '.DS_Store' in file_name:
            continue
                
        ## Make directory for storing results of this dataset
        file_name_no_dir = file_name[file_name.rfind('/')+1:file_name.index('.csv')]
            
        if not os.path.exists('results/' + file_name_no_dir):
            os.mkdir('results/'+ file_name_no_dir)
        else:
            if len(os.listdir('results/' + file_name_no_dir)) > 160:  ## If the dataset has more than 322 models, continue
                continue
           
        print(file_name)
    
        ## Loading the CSV dataset
        try:
            df = pd.read_csv(file_name, error_bad_lines=False, header  = None)
        
            ## Filling NaN values with 0
            df = df.fillna(0)
        
            ## Remove any lines with alphabet texts or special characters 
            df = df[pd.to_numeric(df[0], errors='coerce').notnull()]
        
            ### Remove Outliers
            #q_low = df[0].quantile(0.01)
            #q_hi  = df[0].quantile(0.99)
            #df = df[(df[0] < q_hi) & (df[0] > q_low)]
    
            ## Normalizatio of the dataframe
            x = df.values 
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            df = pd.DataFrame(x_scaled)
            
            
            ## If the dataset is too too short (less than twice prediction length), then skip this dataset; otherwise DEEPAR throws an error
            if len(df) <= 2 * prediction_length:
                continue
        
            cnt += 1
    
    
            ## Iterate over all Lag values
            for lag_par in lag:
                            
                print(len(df))
                
                if len(df) < lag_par: ## No Sufficient Data to Build a model
                    continue
            
                ## Do Expontial_Smoothing on data or not depending on preprocessing option
                        
                for prep in preprocess:
                    if prep != 'No':
                        df_1 = Expontial_Smoothing(df.iloc[:,0], smooth_level)
                        df = pd.DataFrame({"0": df_1})
                   
                    #df = pd.DataFrame({"0": df_1})

                    
                    ## (0) Divide Dataset Into Training and Test depending on Lag Variable Value
                    training_data = ListDataset([{"start": df.iloc[start_ind,0], "target": df.iloc[:end_ind,0]}],freq = freq)
                    test_data = ListDataset([{"start": df.iloc[end_ind,0], "target": df.iloc[:int(end_ind + prediction_length),0]}],freq = freq)
                
    
                    
                    ### (1) DEEPAR Forecasting Model Training and Performance Reporting
            
                    for i in DEEPAR_param_list_1:
                        for j in DEEPAR_param_list_2:
                        
                            ## Start Time of Training the model
                            start_time = time.time()
                    
                            ## Train DeepAR Model
                            estimator = DeepAREstimator(freq = freq, prediction_length = prediction_length, trainer=Trainer(epochs= DEEPAR_epochs), num_cells = i, num_layers = j)
                            predictor = estimator.train(training_data=training_data)
                        
                            ## Save Model Parameters
                            #model_file_name = file"net.params"
                            #net.save_parameters(file_name)
        
                            ## Calculating Run time of the model training process
                            train_run_time_seconds = time.time() - start_time
                            print('DEEPAR Training Time: ' + str(train_run_time_seconds))
                        
                            ## Collecting Relative Performance Metrics
                            inference_run_time_seconds, smape, mape, mse  = Generate_Inference_Metrics(predictor, test_data, Num_Sample_Paths)
                            mse_vec.append(mse)
                            mape_vec.append(mape)
                            smape_vec.append(smape)
                            train_time_vec.append(train_run_time_seconds)
                            infer_time_vec.append(inference_run_time_seconds)
                        
                            print('DEEPAR  Inference Time: ' + str(inference_run_time_seconds))
                            print('DEEPAR MSE: ' +  str(mse))
                        
                            ## Writing the Model Performance File 
                            Write_Performance_File(file_name, "DEEPAR_" + str(i) + '_' + str(j) + '_Preprocess=' + str(prep), lag_par, mse, mape, smape, train_run_time_seconds, inference_run_time_seconds)
                        
                            #print(json.dumps(agg_metrics, indent=4))
                    
                       
                
                    #### (2) Prophet Model Training and Performance Reporting 
                
                    for i in Prophet_param_list_1:
                        for j in Prophet_param_list_2:
                        
                            ## Start Time of Training the model
                            start_time = time.time()
                    
                            # define the model
                            model = Prophet(changepoint_prior_scale = i, seasonality_prior_scale = j)
                    
                            ## Just generate vector of times for Prophet Model to Work
                            base = datetime.datetime(2000, 1, 1)
                            arr = np.array([base + datetime.timedelta(hours=i) for i in range(18)])
                    
                    
                            ## Train Dataset Prepare
                            df_new = pd.DataFrame()
                            df_new['y'] = df.iloc[start_ind: end_ind+1,0]
                            df_new['ds']= arr[:-2]
                    
                            #print(df_new)

                            # fit the model
                            model.fit(df_new)
                    
                            #print(model)
                    
                            ## Calculating Run time of the model training process
                            train_run_time_seconds = time.time() - start_time
                            print('Prophet Training Time: ' + str(train_run_time_seconds))
                    
                            df_test = pd.DataFrame()
                            df_test['ds']= arr[len(arr)-2:len(arr)-1] 
                    
                            #print(df_test)
                            #print(df.iloc[int(lag_par),0])
                                        
                            ## Forecasting using Prophet Model
                    
                            ## Start Time of Prediction
                            start_time = time.time()
                    
                            # Doing Prediction
                            forecast = model.predict(df_test)
                            yhat = forecast['yhat']
                    
                    
                            # Generate its inference metrics
                            mse = mean_square_error(yhat,df.iloc[int(end_ind): int(end_ind) + prediction_length,0])
                            mape = mean_absolute_percentage_error(yhat,df.iloc[int(end_ind):int(end_ind) + prediction_length,0])
                            smape = symm_mean_absolute_percentage_error(yhat,df.iloc[int(end_ind):int(end_ind) + prediction_length, 0])
                            
                            inference_run_time_seconds = time.time() - start_time
                                    
                            print('Prophet Inference Time: ' + str(inference_run_time_seconds))
                            print('Prophet MSE: ' +  str(mse))
                            mse_vec.append(mse)
                            mape_vec.append(mape)
                            smape_vec.append(smape)
                            train_time_vec.append(train_run_time_seconds)
                            infer_time_vec.append(inference_run_time_seconds)
                    
                            ## Writing the Model Performance File 
                            Write_Performance_File(file_name, "Prophet_" + str(i) + '_' + str(j) + '_Preprocess=' + str(prep), lag_par, mse, mape, smape, train_run_time_seconds, inference_run_time_seconds)
                    
                    
                
                
                    #### (3) Seasonal Naive 
                
                    for season_length in Seasonal_Naive_param_list_1:    
                        predictor = SeasonalNaivePredictor(freq, prediction_length, season_length)
            
                        ## Generate its inference metrics
                        inference_run_time_seconds, smape, mape, mse  = Generate_Inference_Metrics(predictor, test_data, Num_Sample_Paths)  
                        print('Seasonal Inference Time: ' + str(inference_run_time_seconds))
                        print('Seasonal MSE: ' +  str(mse))
                        mse_vec.append(mse)
                        mape_vec.append(mape)
                        smape_vec.append(smape)
                        train_time_vec.append(0) ## Seasonal Naive do not do training
                        infer_time_vec.append(inference_run_time_seconds)
                    
                        ## Writing the Model Performance File 
                        Write_Performance_File(file_name, "SeasonalNaive_" + str(season_length) + '_' + '_Preprocess=' + str(prep), lag_par, mse, mape, smape, train_run_time_seconds, inference_run_time_seconds)
            
               
                    #### (4) DeepFactor
            
                    for i in DF_param_list_1:
                        for j in DF_param_list_2:
                
                            ## Start Time of Training the model
                            start_time = time.time()
                            
                            estimator = DeepFactorEstimator(freq = freq, prediction_length= prediction_length, num_hidden_global = i,
                                                        num_factors = j, trainer=Trainer(epochs= DEEPAR_epochs))
                                                                                                
                            predictor = estimator.train(training_data=training_data)
        
                            ## Calculating Run time of the model training process
                            train_run_time_seconds = time.time() - start_time
                            print('DF Training Time: ' + str(train_run_time_seconds))
        
                            ## Generate its inference metrics
                            inference_run_time_seconds, smape, mape, mse  = Generate_Inference_Metrics(predictor, test_data, Num_Sample_Paths)
                            print('DF  Inference Time: ' + str(inference_run_time_seconds))
                            print('DF MSE: ' +  str(mse))
                            mse_vec.append(mse)
                            mape_vec.append(mape)
                            smape_vec.append(smape)
                            train_time_vec.append(train_run_time_seconds)
                            infer_time_vec.append(inference_run_time_seconds)
                        
                            ## Writing the Model Performance File 
                            Write_Performance_File(file_name, "DEEPFactor_" + str(i) + '_' + str(j)+ '_Preprocess=' + str(prep), lag_par, mse, mape, smape, train_run_time_seconds, inference_run_time_seconds)
                        
            
            
                    #### (5) Gaussian Process Estimator
                
                    for i in GP_param_list_1:
                        for j in GP_param_list_2:
            
                            ## Start Time of Training the model
                            start_time = time.time()
            
                            estimator = GaussianProcessEstimator(freq= freq, prediction_length= prediction_length, cardinality = i, max_iter_jitter = j, trainer=Trainer(epochs= DEEPAR_epochs))                    
                            predictor = estimator.train(training_data=training_data)
        
                            ## Calculating Run time of the model training process
                            train_run_time_seconds = time.time() - start_time
                            print('GP Training Time: ' + str(train_run_time_seconds))
        
                            ## Generate its inference metrics
                            inference_run_time_seconds, smape, mape, mse  = Generate_Inference_Metrics(predictor, test_data, Num_Sample_Paths)
                            print('Gaussian Process  Inference Time: ' + str(inference_run_time_seconds))
                            print('Gaussian Process  MSE: ' +  str(mse))
                            mse_vec.append(mse)
                            mape_vec.append(mape)
                            smape_vec.append(smape)
                            train_time_vec.append(train_run_time_seconds)
                            infer_time_vec.append(inference_run_time_seconds)
                        
                            ## Writing the Model Performance File
                            Write_Performance_File(file_name, "GP_" + str(i) + '_' + str(j)+ '_Preprocess=' + str(prep), lag_par, mse, mape, smape, train_run_time_seconds, inference_run_time_seconds)
                       
            
                    #### (6) Autoregression (VAR)
                
                    for i in AR_param_list_1:
                        for j in AR_param_list_2:
                            
                            ## Start Time of Training the model
                            start_time = time.time()
            
                            model = AutoReg(df.iloc[start_ind: end_ind,0], lags= 1, trend = j)
                            model_fit = model.fit(cov_type = i)
            
                            ## Calculating Run time of the model training process
                            train_run_time_seconds = time.time() - start_time
                            print('AR Training Time: ' + str(train_run_time_seconds))
            
                            ## Start Time of inference of the model
                            start_time = time.time() 
                            index = df.index
            
                            # make prediction
                            if prediction_length + int(lag_par) >= len(df):
                                yhat = model_fit.predict(start= index[int(lag_par)], end= len(df), dynamic=True)
                            else:
                                yhat = model_fit.predict(start= index[int(lag_par)], end= index[int(lag_par)+prediction_length-1], dynamic=True)
                        
                    
                            ## Generate its inference metrics (from stats library)                            
                            mse = mean_square_error(yhat,df.iloc[int(end_ind): int(end_ind) + prediction_length,0])
                            mape = mean_absolute_percentage_error(yhat,df.iloc[int(end_ind):int(end_ind) + prediction_length,0])
                            smape = symm_mean_absolute_percentage_error(yhat,df.iloc[int(end_ind):int(end_ind) + prediction_length, 0])
                            
                            inference_run_time_seconds = time.time() - start_time
                    
                            print('Auto Regression Inference Time: ' + str(inference_run_time_seconds))
                            print('Auto Regression  MSE: ' +  str(mse)) 
                            mse_vec.append(mse)
                            mape_vec.append(mape)
                            smape_vec.append(smape)
                            train_time_vec.append(train_run_time_seconds)
                            infer_time_vec.append(inference_run_time_seconds)
                    
                            ## Writing the Model Performance File
                            Write_Performance_File(file_name, "AR_" + str(i) + '_' + str(j)+ '_Preprocess=' + str(prep), lag_par, mse, mape, smape, train_run_time_seconds, inference_run_time_seconds)
                    
                    
                    ####  (7) Random Forests Model
                
                    ## Train Data Preparation
                    train_data = df.iloc[start_ind-1: int(end_ind)+1,0]
                
                    # transform the time series data into supervised learning
                    train = series_to_supervised(train_data, n_in = lag_par)
                    # split into input and output columns
                    trainX, trainy = train[:, :-1], train[:, -1]
                
                    #
                    for i in RF_param_list_1:
                        for j in RF_param_list_2: 
                
                            ## Start Time of Training the model
                            start_time = time.time()
            
                            ## fit model
                            if j!= 'None':
                                model = RandomForestRegressor(n_estimators= i, max_depth = j)
                            else:
                                model = RandomForestRegressor(n_estimators= i)
                                
                            model.fit(trainX, trainy)
                
                            ## Calculating Run time of the model training process
                            train_run_time_seconds = time.time() - start_time
                            print('RF Training Time: ' + str(train_run_time_seconds))
                
                
                            # construct an input for a new prediction
                            row = df.iloc[start_ind: int(end_ind)+1,0] #train_data[-lag_par-1:]
                            test = df.iloc[int(end_ind)+2,0]
                            
                            # make a one-step prediction
                            yhat = model.predict(asarray([row]))
                            #print('Input: %s, Predicted: %.3f' % (row, yhat[0]))
                
                            ## Generate its inference metrics (from stats library)
                            mse = mean_square_error(yhat,test)
                            mape = mean_absolute_percentage_error(yhat,test)
                            smape = symm_mean_absolute_percentage_error(yhat,test)
                            inference_run_time_seconds = time.time() - start_time
                
                            print('Random Forest Inference Time: ' + str(inference_run_time_seconds))
                            print('Random Forest MSE: ' +  str(mse)) 
                            mse_vec.append(mse)
                            mape_vec.append(mape)
                            smape_vec.append(smape)
                            train_time_vec.append(train_run_time_seconds)
                            infer_time_vec.append(inference_run_time_seconds)
                
                            ## Writing the Model Performance File
                            Write_Performance_File(file_name, "RF_" + str(i) + '_' + str(j)+ '_Preprocess=' + str(prep), lag_par, mse, mape, smape, train_run_time_seconds, inference_run_time_seconds)
                
                
                           
            
 #           if cnt >= 3:
 #               break
            print(cnt)
    
        except: ## if the file is corrupt or empty skip dataset
            continue
        


