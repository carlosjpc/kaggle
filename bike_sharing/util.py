import pandas as pd
import numpy as np
from zlib import crc32
import matplotlib.pyplot as plt
import seaborn as sns

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
    
