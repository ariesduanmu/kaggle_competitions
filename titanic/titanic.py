import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import Imputer

def fill_nan_age_col(data):
    age_avg = data['Age'].mean()
    age_std = data['Age'].std()
    age_null_count = data['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    data['Age'][np.isnan(data['Age'])] = age_null_random_list
    data['Age'] = data['Age'].astype(int)
    return data

def replace_age_to_group_num(data):
    data.loc[data["Age"] <= 16, 'Age'] = 0
    data.loc[(data["Age"] > 16) & (data['Age'] <= 32), 'Age'] = 1
    data.loc[(data["Age"] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data["Age"] > 48) & (data['Age'] <= 64), 'Age'] = 3
    data.loc[(data["Age"] > 64), 'Age'] = 4
    return data

def replace_fare_to_group_num(data):
    data.loc[data["Fare"] <= 7.91, 'Fare'] = 0
    data.loc[(data["Fare"] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data["Fare"] > 14.454) & (data['Fare'] <= 31), 'Fare'] = 2
    data.loc[(data["Fare"] > 31), 'Fare'] = 3
    data['Fare'] = data['Fare'].astype(int)
    return data

def replace_sex_to_num(data):
    data["Sex"] = data["Sex"].map({'female':0,'male':1}).astype(int)
    return data

def replace_embarked_to_num(data):
    data["Embarked"] = data["Embarked"].map({'S':0, 'C':1, 'Q':2}).astype(int)
    return data

def sum_up_sibsp_parch(data):
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    return data

def fill_embarked_nan(data):
    data["Embarked"] = data["Embarked"].fillna('S')
    return data

def fill_fare_nan(data):
    data['Fare'] = data['Fare'].fillna(train['Fare'].median())
    return data
def read_data(data_path):
    train = pd.read_csv("train.csv")

def get_random_forest_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    maes = []
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        val = [1 if p >= i else 0 for p in preds_val]
        mae = mean_absolute_error(targ_val, val)
        maes += [mae]
    return(maes)

def clean_data(train):
    train['Has_Cabin'] = train['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
    train = train.drop(["PassengerId","Name","Ticket","Cabin"], axis=1)

    for data in train: 
        data = replace_sex_to_num(data)
        data = sum_up_sibsp_parch(data)
        data = fill_nan_age_col(data)
        data = fill_fare_nan(data)

    for data in train:
        data = replace_embarked_to_num(data)
        data['IsAlone'] = 0
        data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
        
    for data in train:
        data = replace_age_to_group_num(data)
        data = replace_fare_to_group_num(data)
    return train

def validate(train_X, val_X, train_y, val_y):
    for max_leaf_nodes in [5, 50, 500, 5000,]:
        random_forest_maes = get_random_forest_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
        print("Max leaf nodes(random forest): %d  \t\t Mean Absolute Error:  %s" %(max_leaf_nodes, random_forest_maes))

def let_it_work(train):
    predictors = ["Sex","Pclass","Age","Fare","Embarked","Has_Cabin","FamilySize"]
    X = train[predictors]
    y = train.Survived
    train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
    validate(train_X, val_X, train_y, val_y)

if __name__ == "__main__":
    data_path = "train.csv"
    data = read_data(data_path)
    data = clean_data(data)
    let_it_work(data)


