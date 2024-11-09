#  Probably should separate

# General Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sklearn packages
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report



def version_control(file_path):

    try:
        with open(file_path, 'r') as file:
            count = int(file.read().strip())
    except FileNotFoundError:
        count = 0

    count += 1

    with open(file_path, 'w') as file:
        file.write(str(count))

    return count


def multiencoder(dataset:pd.DataFrame, features:list, encoder_type:str, encoders_dict:dict = {}) -> (dict):
    
    if encoder_type == "binary":
        
        for feature in features:
            le = LabelEncoder()
            dataset[feature] = le.fit_transform(dataset[feature])
            encoders_dict[feature] = le
    
    elif encoder_type == "frequency":
            
        for feature in features:  
            freq_encoding = dataset[feature].value_counts(normalize=True)
            dataset[feature] = dataset[feature].map(freq_encoding)
            encoders_dict[feature] = {val: float(freq) for val, freq in zip(freq_encoding.index, freq_encoding.values)}
    
    else:
        print("Invalid Encoder")
    return
        
    #return encoders_dict


def gen_dummy(dataset:pd.DataFrame, features:list) -> None:
    for feature in features:
        dataset[feature + " Occurred"] = dataset[feature].notna().astype(int)
    return


def date_past_accident(dataset:pd.DataFrame, features:list) -> None:
    for feature in features:
        new_name = feature.replace("Date","")
        new_name = new_name.replace(" ","")
        dataset[f"{new_name} Years past Accident"] = (pd.to_datetime(dataset[feature], errors='coerce') - dataset["Accident Date"]).dt.days / 365.25


