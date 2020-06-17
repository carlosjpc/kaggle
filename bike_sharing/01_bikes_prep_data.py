import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from sklearn.decomposition import PCA
from util import DatasetType
import os

# Function that should be imported when dealing with new data
def bikes_full_pipeline(data,dataset_type,folder,drop_first=True):
    dt_col = "datetime"
    target_name = "totalRides"
    target_rel_names = ["registered","casual"] 
    data_scaled, data_prepared = bikes_pipeline(data, dt_col, target_name, target_rel_names,dataset_type, folder, drop_first)
    if dataset_type == DatasetType.PREP:
        return data_prepared
    return data_scaled

def bikes_pipeline(data, dt_col, target_name, target_rel_names, dataset_type, folder, drop_first):
    data = modify_features(data, dt_col, target_name, target_rel_names)
    data_num, data_cat, data_target = create_bikes_num_cat_target(data, dt_col, target_name)
    data_num_scaled = scale_data(data_num,dataset_type,folder)
    data_cat_dummies = pd.get_dummies(data_cat.astype(str), drop_first=drop_first)
    data_prepared = pd.concat([data_num,data_cat_dummies,data_target], axis=1)
    data_scaled = pd.concat([data_num_scaled,data_cat_dummies,data_target], axis=1)
    return data_scaled, data_prepared
    
def scale_data(data,dataset_type,folder):
    sc=StandardScaler() 
    sc.fit(data.to_numpy())
    data_scaled =sc.transform(data.to_numpy())
    joblib.dump(sc, os.path.join(folder,"SC_"+dataset_type.value+".pkl"))
    return pd.DataFrame(data=data_scaled, columns=data.columns)

    
def modify_features(data, dt_col, target_name, target_rel_names):
    data = delete_features(data , target_rel_names)
    data.rename(columns={"count": target_name}, inplace=True)
    data = add_datetime_columns(data,dt_col)
    return data
       
def add_datetime_columns(data,dt_col):
    if dt_col in data.columns:
        data["hour"]      = data[dt_col].dt.hour
        data['dayOfWeek'] = data[dt_col].dt.dayofweek
        data['month']     = data[dt_col].dt.month
        data['year']      = data[dt_col].dt.year
        data['time']      = data[dt_col].dt.year+bikes[dt_col].dt.month+bikes[dt_col].dt.dayofweek+bikes[dt_col].dt.hour
    return data
        
def create_bikes_num_cat_target(data, dt_col, target_name):
    data_target = data[target_name]
    cat_features = ['weather','season','holiday','workingday'] #year, hour??
    num_features = del_from_list(list(data.columns),cat_features)
    num_features.remove(dt_col)
    num_features.remove(target_name)
    data_num , data_cat = (data[num_features] , data[cat_features])
    return data_num, data_cat, data_target

def delete_features(data , features_to_delete):
    for feature_to_delete in features_to_delete:
        if feature_to_delete in data.columns:
            data.drop(feature_to_delete, axis=1, inplace=True)
    return data 

def del_from_list(list_, unwanted_elements):
    for element in unwanted_elements:
        if element in list_:
            list_.remove(element)
    return list_

def train_test_valid(X,y,test_size=.15,train_size=.15, random_state=42, create_valid=True):
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, 
                                                        test_size=test_size, random_state=42)
    
    if(create_valid):
        val_size = int(len(X_train_full) * train_size)    
        X_valid, X_train = X_train_full[:val_size] , X_train_full[val_size:]
        y_valid, y_train = y_train_full[:val_size], y_train_full[val_size:]      
        return (X_train, y_train, X_test, y_test, X_valid, y_valid)
    
    return (X_train_full, y_train_full, X_test, y_test) 

def fill_wind(model,data,
              windColumns = ["season","weather","humidity","month","temp","year","atemp"]):
    dataWind0 = data[data["windspeed"]==0]
    dataWindNot0 = data[data["windspeed"]!=0]
    wind0Values = model.predict(X= dataWind0[windColumns])
    dataWind0["windspeed"] = wind0Values
    data_with_wind = dataWindNot0.append(dataWind0)
    return data_with_wind.sort_index()
    
if __name__ == '__main__':
    dt_col = "datetime"
    target_name = "totalRides"
    target_rel_names = ["registered","casual"] 
    path = "dataset/prepared_data_and_models"
    bikes = pd.read_csv('dataset/train.csv', parse_dates = ['datetime'])

    from sklearn.ensemble import RandomForestRegressor
    
    bikes_mod = modify_features(bikes.copy(), dt_col, target_name, target_rel_names)
    dataWind0 = bikes_mod[bikes_mod["windspeed"]==0]
    dataWindNot0 = bikes_mod[bikes_mod["windspeed"]!=0]
    rfModel_wind = RandomForestRegressor()
    windColumns = ["season","weather","humidity","month","temp","year","atemp"]
    rfModel_wind.fit(dataWindNot0[windColumns], dataWindNot0["windspeed"])
    joblib.dump(rfModel_wind,path+"/regressor_model_for_wind.pkl")
    
    bikes_with_wind = fill_wind(rfModel_wind,bikes_mod.copy())
    bikes["windspeed"] = bikes_with_wind["windspeed"]
    
    for dataset_type in DatasetType:
        data = bikes_full_pipeline(bikes,dataset_type,folder=path) 
        X = data.drop(target_name,axis=1)
        y = data[target_name]
        if dataset_type == DatasetType.PCA:
            pca = PCA(.99)
            X = pca.fit_transform(X)
            joblib.dump(pca, path+"/pca_fitted.pkl")
        elif dataset_type == DatasetType.SCALED21:
            data = bikes_full_pipeline(bikes,dataset_type,folder=path,drop_first=True)
        Xy = train_test_valid(X, y, create_valid=False)
        joblib.dump(Xy, path+"/final/Xy_"+dataset_type.value+".pkl")
        Xy = train_test_valid(X, y, create_valid=True)
        joblib.dump(Xy, path+"/train/Xy_"+dataset_type.value+".pkl")
        
    
    
    