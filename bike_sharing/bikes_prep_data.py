import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib


# Function that should be imported when dealing with new data
def bikes_full_pipeline(data):
    dt_col = "datetime"
    target_name = "totalRides"
    target_rel_names = ["registered","casual"] 
    data_scaled, data_prepared = bikes_pipeline(data, dt_col, target_name, target_rel_names)
    return data_scaled, data_prepared

def bikes_pipeline(data, dt_col, target_name, target_rel_names):
    data = modify_features(data, dt_col, target_name, target_rel_names)
    data_num, data_cat, data_target = create_bikes_num_cat_target(data, dt_col, target_name)
    data_num_scaled = scale_data(data_num)
    data_cat_dummies = pd.get_dummies(data_cat.astype(str), drop_first=True)
    data_prepared = pd.concat([data_num,data_cat_dummies,data_target], axis=1)
    data_scaled = pd.concat([data_num_scaled,data_cat_dummies,data_target], axis=1)
    return data_scaled, data_prepared

def scale_data(data):
    sc=StandardScaler() 
    sc.fit(data.to_numpy())
    data_scaled =sc.transform(data.to_numpy())
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


if __name__ == '__main__':
    target_name = "totalRides"
    bikes = pd.read_csv('dataset/train.csv', parse_dates = ['datetime'])
    data_scaled, data_prepared = bikes_full_pipeline(bikes)
    
    # Should a time series be randomly splitted ? 
    X=data_scaled.drop(target_name,axis=1)
    y=data_scaled[target_name]
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, 
                                                        test_size=0.15, random_state=42)
    val_size = len(X_train_full) // 7
    
    X_valid, X_train = X_train_full[:val_size] , X_train_full[val_size:]
    y_valid, y_train = y_train_full[:val_size], y_train_full[val_size:]
    
    XY = (X_train, X_test, y_train, y_test, X_valid, y_valid)
    joblib.dump(data_scaled, "dataset/data_scaled.pkl")
    joblib.dump(XY, "dataset/XY.pkl")

    