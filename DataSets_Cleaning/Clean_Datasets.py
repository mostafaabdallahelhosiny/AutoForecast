import pandas as pd
import os
import simplejson as json
import csv
import numpy as np

'''
##### TPCPD Datasets Cleaning and Processing
dirs_TPCPD = os.listdir('data-raw/TCPD_datasets/')
print(dirs_TPCPD)


count = 0
for dir in dirs_TPCPD:
    print(dir)
    if '.DS_Store' not in dir:
        files_TPCPD = os.listdir('data-raw/TCPD_datasets/'+dir)
        print(files_TPCPD)
    
        for file in files_TPCPD:
                if '.json' in file:
                    count += 1
                    with open('data-raw/TCPD_datasets/'+dir+'/'+file) as train_file:
                        dict_train = json.load(train_file)
                    

                        # converting json dataset from dictionary to dataframe
                        train = pd.DataFrame.from_dict(dict_train, orient='index')
                        train.reset_index(level=0, inplace=True)
                        series  = train.iloc[5,:][0][0]['raw']
                        print(series)
                        
                        wtr = csv.writer(open ('data-cleaned/TCPD_datasets/'+file.split('.json')[0]+'.csv', 'w'), delimiter=',', lineterminator='\n')
                        for x in series : wtr.writerow ([x])

print(count)
print(len(dirs_TPCPD))


###### MHSETS Datasets Cleaning and Processing
dirs_MHSETS = os.listdir('data-raw/MHSETS/')
print(dirs_MHSETS)

count = 0

for dir in dirs_MHSETS:
    print(dir)
    if 'READ' in dir:
        continue
    if '.DS_Store' not in dir:
        files_MHSETS = os.listdir('data-raw/MHSETS/'+dir)
        print(files_MHSETS)   
        
        for file in files_MHSETS:
            #print(file)
            if '.DS_Store' not in file:
                arr = []
                f = open('data-raw/MHSETS/'+dir+'/'+file, "rt")
            
                csvfile = open('data-cleaned/MHSETS/'+file + '.csv', 'w',newline=None) 
            
                count = 0
            
                while True:
                    count += 1
 
                    # Get next line from file
                    line = f.readline()
                
                    if count > 1:
                        for num in line.split(' '):
                            if num and num!= '\n':
                                csvfile.writelines(num+'\n')
 
                    # if line is empty # end of file is reached
                    if not line:
                        break
                        print("Line{}: {}".format(count, line.strip()))
 
                f.close()
    
                   

##### Cleaning Mustafa's Previous Dataset
dirs_Mus = os.listdir('data-raw/Mustafa Prev time-series datasets/')
print(dirs_Mus)
    
for file in dirs_Mus:
        if 'temp' not in file and '.DS_Store' not in file:
            print (file)
            col_list = ["x"]
            df = pd.read_csv('data-raw/Mustafa Prev time-series datasets/'+file)
            
            ## Writing the three rotor sensors time-series to CSV files
            wtr_x = csv.writer(open ('data-cleaned/Mustafa Prev time-series datasets/'+file.split('.csv')[0]+'_X.csv', 'w'), delimiter='\n', lineterminator='\n')
            wtr_x.writerow (df.iloc[:,0].T)
            
            wtr_y = csv.writer(open ('data-cleaned/Mustafa Prev time-series datasets/'+file.split('.csv')[0]+'_Y.csv', 'w'), delimiter='\n', lineterminator='\n')
            wtr_y.writerow (df.iloc[:,1].T)
            
            
            wtr_z = csv.writer(open ('data-cleaned/Mustafa Prev time-series datasets/'+file.split('.csv')[0]+'_Y.csv', 'w'), delimiter='\n', lineterminator='\n')
            wtr_z.writerow(df.iloc[:,2].T)
'''           

'''
dirs_univar = os.listdir('data-types-final/Uni_Variate/')
i = 0
for file_name in dirs_univar:
    i += 1
    df = pd.read_csv('data-types-final/Uni_Variate/' + file_name, error_bad_lines=False, header  = None)
    print(str(i) + ' & ' + file_name.split('.csv')[0] + ' & ' + str(len(df)) + ' & ' + 'Uni-var\\' )
'''

i = 0
j = 0
dirs_others = os.listdir('data-types-orig/Multi_Variate_Real/')
for dir_name in dirs_others:
    if '.DS_Store' in dir_name:
        continue
    i += 1
    files_list_dir = os.listdir('data-types-orig/Multi_Variate_Real/'+ dir_name )
    
    for file_name in files_list_dir:
        if '.DS_Store' not in file_name:
            j += 1
            df = pd.read_csv('data-types-orig/Multi_Variate_Real/' + dir_name + '/' + file_name, error_bad_lines=False, header  = None)
            print(str(i) + ' & ' + dir_name  + ' & ' + file_name.split('.csv')[0] + ' & ' + str(len(df)) +  ' &' )
print(dirs_others)


'''
## Cleaning Other Datasets
dirs_others = os.listdir('data-raw/')
print(dirs_others)


for file in dirs_others:
    df = pd.read_csv('data-raw/'+file)
    print(file) 
    
    print(df)
    
    #df = df.replace(',','', regex=True)
    df = df.replace(';',' ', regex=True)
   
    
    print(df) 
    print(df.columns)
    
    
    #print(df.iloc[:,1])       

    for col in df.columns:
        if col != 'month' and col != 'id' and col != 'date' and 'Date' not in col and col != 'hour' and col != 'day' and col != 'year' and col != 'label' and col != 'station' and col != 'No':
            wtr = csv.writer(open ('New/'+ file.split('.csv')[0] + '_' +str(col.split('[')[0])+'.csv', 'w'), delimiter='\n', lineterminator='\n')
            wtr.writerow(df[col])
'''


'''
## Cleaning Sales Transaction Data (Has Rows as products and columns as time-series data)
df = pd.read_csv('Sales_Transactions_Dataset_Weekly.csv')
print(df.head())
print(df.columns)
print(len(df))

for i in range(len(df)):
            wtr = csv.writer(open ('Multi_Variate_Real/Sales_Transactions/' + 'p' + str(i+1) +'.csv', 'w'), delimiter='\n', lineterminator='\n')
            wtr.writerow(df.iloc[i,1:53])

'''












