import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.impute import KNNImputer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor
from pathlib import Path
import pickle
import re

df = pd.read_csv("/home/dibepa/git/global.agb.ml/data/predict/predictor.data.onlybioclim/predictors_global_data_lon_-72.0_lat_-18.0.csv")

pickled_model = "/home/dibepa/git/global.agb.ml/data/training/predictor_selection_onlybioclim/abd_model_onlybioclim.pkl"


df1 = df[ (df.x < -62) & (df.x > -64) & (df.y > -20)  ]

df2 = df[ (df.x < -60) & (df.x > -62) & (df.y > -20)  ] 


class BGRBinaryEncoding(BaseEstimator, TransformerMixin):
    def __init__(self):
        self
    def fit(self, X , y=None):
        return self
    def transform(self, X):
        df = X.copy()
        df["bgr_tuple"] = [binary_encoding(t) for t in df.bgr] 
        try:
            df["bgr1"] , df["bgr2"], df["bgr3"] = df.bgr_tuple.str
        except:
            df["bgr1"] , df["bgr2"], df["bgr3"] = (np.nan,np.nan,np.nan)
            df = df.dropna()

        df = df.drop(["bgr_tuple","bgr"],axis="columns")
        return df
    
# custom transformer for sklearn pipeline
class ColumnExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        return X[self.cols]

    def fit(self, X, y=None):
        return self    
    
class LogarithmizeWaterObservables(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
    def fit(self, X , y=None):
        return self
    def transform(self, X):
        df = X.copy()
        log_preds = log_predictors(self.cols)
        df[log_preds] = np.log(df[log_preds]+1)     
        return df    

def keep_predictor(plist,pname,pval):
    if pval>0.5:
        plist.append(pname)
    return plist


def build_predictor_list(predictor_dict):
    plist = []
    for predictor in predictor_dict:
        plist = keep_predictor(plist,predictor,predictor_dict[predictor])
    return plist    

def func(x):
    return np.log(x)

def inverse_func(x):
    return np.exp(x)

def binary_encoding(bgr):
    match bgr:
        case "Palearctic":
            return (0,0,0)
        case "Indomalayan":
            return (0,1,0)
        case "Australasia":
            return (0,0,1)
        case "Nearctic":
            return (0,1,1)
        case "Afrotropic":
            return (1,0,0)
        case "Neotropic":
            return (1,1,0)  
        case _:
            return np.nan               

log_transformer = FunctionTransformer(func=func,inverse_func=inverse_func)

def log_predictors(predictor_list):
    all = ["yp","pwm","pet","ps","pcq","pdq","pwaq","pweq"]
    return [f for f in all if f in predictor_list]


def predictor_list(predictor_string):
    pred = predictor_string.split("-")
    return pred 


#############################################################################


with open(pickled_model, 'rb') as f:
    model = pickle.load(f)

for df in [df1,df2]:


    coords = df[["x","y"]]
    X = df.drop(["x","y"],axis="columns")

    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    try:

        abd= model.predict(X)

        abd_df = pd.DataFrame({"abd":abd})
        print(abd_df)
        
        predicted = pd.concat([abd_df,coords], axis = "columns")

        # print(predicted)    
    except:
        print("Could not make prediction.")
        continue