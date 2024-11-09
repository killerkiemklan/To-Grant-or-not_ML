# General Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sklearn packages
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import RFE


# embedded methods
from sklearn.linear_model import LassoCV

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score

import scipy.stats as stats
from scipy.stats import chi2_contingency

def TestIndependence(X,y,var,alpha=0.05):
    dfObserved = pd.crosstab(y,X)
    chi2, p, dof, expected = stats.chi2_contingency(dfObserved.values)
    dfExpected = pd.DataFrame(expected, columns=dfObserved.columns, index = dfObserved.index)
    if p<alpha:
        result="{0} is IMPORTANT for Prediction".format(var)
    else:
        result="{0} is NOT an important predictor. (Discard {0} from model)".format(var)
    print(result)


def feature_selection_RFE(X,y,n_features,model=None):
    best_score = 0
    best_features = []

    results = {}
    
    results = {}
    
    for feature in range(1,n_features):
        
        rfe = RFE(estimator=model, n_features_to_select=feature)
        rfe.fit(X, y)

        selected_features = X.columns[rfe.support_]
        
        y_pred = rfe.predict(X)
        
        macro_f1 = f1_score(y, y_pred, average='macro')
        
        results[feature] = selected_features
        
        if macro_f1 > best_score:
            best_score = macro_f1
            best_features = selected_features.tolist()  
    
    return best_features


def feature_selection_Lasso(X,y):
    reg = LassoCV()
    reg.fit(X, y)
    coef = pd.Series(reg.coef_, index = X.columns)
    coef.sort_values()
    plot_importance(coef,'Lasso')
    print(coef)


def plot_importance(coef,name):
    imp_coef = coef.sort_values()
    plt.figure(figsize=(8,10))
    imp_coef.plot(kind = "barh")
    plt.title("Feature importance using " + name + " Model")
    plt.show()