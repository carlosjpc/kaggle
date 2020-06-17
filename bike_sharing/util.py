import pandas as pd
import numpy as np
from zlib import crc32
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from enum import Enum
import tensorflow as tf

class DatasetType(Enum):
    PREP = 'prep'
    SCALED = 'scaled'
    PCA = 'pca'
    SCALED21 = 'scaled21'

def print2(df, colWidth=10, numCols = None, display_width=400):  
    setPdDisplayOptions(len(df), colWidth, numCols, display_width)
    printWithUpperLowerBreakLines(df.rename(columns=lambda x: x[:colWidth - 1] + '/' if len(x) > colWidth else x))
    resetPdDisplayOptions()
    
def setPdDisplayOptions(maxRows, colWidth, numCols, display_width):
    pd.set_option('display.max_colwidth', colWidth)
    pd.set_option('expand_frame_repr', True)
    pd.set_option('precision', 2)
    pd.set_option('display.max_rows', maxRows)
    pd.set_option('display.max_columns', numCols)
    pd.set_option('display.width', display_width)
    
def printWithUpperLowerBreakLines(x):
    print("")
    print(x)
    print("")
    
def resetPdDisplayOptions():
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')
    pd.reset_option('precision')
    pd.reset_option('expand_frame_repr')
    
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

def compareCategoryProportionsSamples(originalData, colName, randomData, stratifiedData=None):
    if colName in originalData.columns and colName in randomData.columns:
        originalProportions = originalData[colName].value_counts() / len(originalData)
        randomProportions = randomData[colName].value_counts() / len(randomData)
        df = pd.DataFrame()
        df["originalProportions"] =  originalProportions
        df["randomProportions"] =  randomProportions
        if stratifiedData is not None and colName in stratifiedData.columns:
            stratifiedProportions = stratifiedData[colName].value_counts() / len(stratifiedData)
            df["stratifiedProportions"] =  stratifiedProportions
        return df
    
def ascendingCorrelation(df, colName):
    corr_matrix = df.corr()
    corr_target = corr_matrix[colName].sort_values(ascending=False)
    mostCorrelatedVarNames = list(corr_target.index)
    return corr_matrix, corr_target, mostCorrelatedVarNames

def display_scores(scores):
    print("")
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    
def plot_pca_with_hue(pca_data, hue, rot=0):
    palette=sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True, rot=rot)
    ax = plt.subplots()
    ax = sns.scatterplot(x=pca_data.iloc[:,0], y=pca_data.iloc[:,1], hue=hue, data=pca_data, palette=palette, alpha=0.5)
    ax.set_xlabel('pca 1')
    ax.set_ylabel('pca 2')
    if len(pca_data.columns) == 3:
        ax = plt.subplots()
        ax = sns.scatterplot(x=pca_data.iloc[:,1], y=pca_data.iloc[:,2], hue=hue, data=pca_data, palette=palette, alpha=0.5)
        ax.set_xlabel('pca 2')
        ax.set_ylabel('pca 3')
        ax = plt.subplots()
        ax = sns.scatterplot(x=pca_data.iloc[:,0], y=pca_data.iloc[:,2], hue=hue, data=pca_data, palette=palette, alpha=0.5)
        ax.set_xlabel('pca 1')
        ax.set_ylabel('pca 3')

def pca_col_names(n_cols):
    col_names =[]
    for i in range(n_cols):
        col_names.append("pca_"+str(i+1))
    return col_names

def describe_Xy_data(X_train, y_train, X_test, y_test, X_valid=None,y_valid=None):
    print('DATA DESCRIPTION')
    print('features     : ' + str(X_train.shape[1]))
    print('y range      : ' + str(min(y_train)) +' - ' + str(max(y_train)))
    print('train samples: ' + str(X_train.shape[0]))
    if X_valid is not None:
        print('valid samples: ' + str(X_valid.shape[0]))
    print('test samples : ' + str(X_test.shape[0]))

def reshape_array(var):
    if type(var) is not np.ndarray:
        var = var.to_numpy()    
    return var.reshape(-1,1)
        
def test_rmsle(model, X_test, y_test, name='Model',print_=True, ylog_model=False):
    y_pred = reshape_array(model.predict(X_test)) #np.maximum(reshape_array(model.predict(X_test)),0)
    if ylog_model:
        y_pred = np.exp(y_pred)
    y_test = reshape_array(y_test)
    sub = y_pred-y_test
    perc=abs(sub)/y_test*100
    error =  rmsle(y_pred, y_test)
    if print_:
        print("\n{} MSE: {:.3f} ~= {:.1f}%".format(name, error, perc.mean()))
    return error, perc.mean()

def rmsle(y, y_):
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))

def tf_rmsle(y, y_):
    return tf.math.sqrt(tf.keras.losses.MeanSquaredLogarithmicError(y,y_))
            
def return_list_Xy(folders):
    path = os.path.join(*folders)
    Xy_files = []
    for file in os.listdir(path):
        if "Xy" in file:
            Xy_files.append(file.replace('.pkl',''))
    return Xy_files

def load_dataset(dataset_type,folder):
    Xy_files = return_list_Xy(folder)
    for file in Xy_files:
        if dataset_type.name.lower() in file.replace('Xy_','').lower():
            print(file + ' loaded\n-------')
            return joblib.load(os.path.join(*folder,file+'.pkl'))
        
def test_models(models_folder,data_folder,ylog_model=False):
    mses = []
    percs = []
    names = []
    result = pd.DataFrame()
    Xy_files = return_list_Xy(data_folder)
    for model_file in os.listdir(os.path.join(*models_folder)):
        has_data = False
        for Xy_file in Xy_files:
            if Xy_file.replace('Xy_','') in model_file.lower():
                data =  joblib.load(os.path.join(*data_folder,Xy_file+'.pkl'))
                X_test, y_test = (data[2], data[3])
                has_data = True
                break
        if has_data:  
            mse, perc = (0,0)
            if model_file.endswith(".h5"):
                model = keras.models.load_model(os.path.join(*models_folder,model_file))
                mse, perc = test_rmsle(model,X_test,y_test,model_file.replace(".h5",""), print_=False,ylog_model=ylog_model)                
            if model_file.endswith(".pkl"):
                model = joblib.load(os.path.join(*models_folder,model_file))
                mse, perc = test_rmsle(model,X_test,y_test,model_file.replace(".pkl",""), print_=False,ylog_model=ylog_model)
            mses.append(mse)
            percs.append(perc)
            names.append(model_file)
        else:
            print('no data found for model: ' + model_file)
    result['model'] = names
    result['rmsle']   = mses
    result['perc']  = percs
    return result
            
if __name__ == '__main__':
    
    DATA_FOLDER = ['dataset', 'prepared_data_and_models', 'train']
    skmodels = test_models(models_folder=['SKmodels','ylog'],data_folder=DATA_FOLDER)
    tfmodels = test_models(models_folder=['TFmodels','ylog'],data_folder=DATA_FOLDER)
    
    result_tests = pd.concat([skmodels,tfmodels])
    result_tests.sort_values(by=['rmsle'],inplace=True)
    print(result_tests)
            