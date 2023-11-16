import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class BGRBinaryEncoding(BaseEstimator, TransformerMixin):
    def __init__(self):
        self
    def fit(self, X , y=None):
        return self
    def transform(self, X):
        df = X.copy()
        df["bgr_tuple"] = [binary_encoding(t) for t in df.bgr] 
        df["bgr1"] , df["bgr2"], df["bgr3"] = df.bgr_tuple.str
        df = df.drop(["bgr_tuple","bgr"],axis="columns")
        # print(df)   
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
        # print(df)   
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

log_transformer = FunctionTransformer(func=func,inverse_func=inverse_func)

def log_predictors(predictor_list):
    all = ["yp","pwm","pet","ps","pcq","pdq","pwaq","pweq"]
    return [f for f in all if f in predictor_list]
   
# Read data from the corresponding outer fold.
fold = 10
data = pd.read_csv("/home/dibepa/git/global.agb.ml/data/training/preprocessed_onlybioclim/outcvloop_train_fold_{}.csv".format(fold))


# test:
yt=1
tdq=1
twq=0
tcq=0
ts=0
yp=1
pwm=1
pet=0
iso=1
ps=0
mtwm=1
bgr=1
mdr=0
mtwq=0
mtcm=0
pcq=0
pdm=0
pdq=0
pwaq=0
pweq=0
tar=0

seed = 1
md = 2
e=0.2
mcw=1
subsample=0.7
g=5
mds = 1


# Dictionary of predictors.
predictor_dict = {
    "yt" : yt,
    "tdq" : tdq,
    "twq" : twq,
    "tcq" : tcq,
    "ts" : ts,
    "yp" : yp,
    "pwm" : pwm,
    "pet" : pet,
    "iso" : iso,
    "ps" : ps,
    "mtwm" : mtwm,
    "mdr" : mdr,
    "bgr" : bgr,
    "mtwq" : mtwq,
    "mtcm" : mtcm,
    "pcq" : pcq,
    "pdm" : pdm,
    "pdq" : pdq,
    "pwaq": pwaq,
    "pweq": pweq,
    "tar": tar
}


if seed<0:
    seed = -1*seed

# Extract the target and predictors.
predictor_list = build_predictor_list(predictor_dict)
npredictors = len(predictor_list)

if npredictors > 0:

    n_splits = 5
    cvs = np.zeros(n_splits)
    
    mdint = int(md)
    
    bst = XGBRegressor(
            n_estimators=100,
            learning_rate=e,
            max_depth = mdint,
            min_child_weight=mcw,
            subsample=subsample,
            min_split_loss = g,
            max_delta_step = mds,
            random_state = seed,
            eval_metric = "rmse",
            objective='reg:squarederror',
            )
    
    # Log-transform target observable.
    regr = TransformedTargetRegressor(regressor=bst,func=func,inverse_func=inverse_func)

    # Log-transform precipitation related variables and binarize biogeographic realm.
    log_preds = log_predictors(predictor_list)
    if len(log_preds)>0:
        if "bgr" in predictor_list:
            estimator = Pipeline([
                ("col_extract", ColumnExtractor(predictor_list)),
                ("bgr_binary", BGRBinaryEncoding()),
                ("log_water", LogarithmizeWaterObservables(predictor_list)),
                ("regressor",regr)
            ])
        else:   
            estimator = Pipeline([
                ("col_extract", ColumnExtractor(predictor_list)),
                ("log_water", LogarithmizeWaterObservables(predictor_list)),
                ("regressor",regr)
            ])
    else:
        if "bgr" in predictor_list:
            estimator = Pipeline([
                ("col_extract", ColumnExtractor(predictor_list)),
                ("bgr_binary", BGRBinaryEncoding()),
                ("regressor",regr)
            ])
        else:
           estimator = Pipeline([
                ("col_extract", ColumnExtractor(predictor_list)),
                ("regressor",regr)
            ])
        
    # print("Starting.")
     
    # Set up 5 folds for inner cross-validation with 1 repeat because the evolutionary algorithm will resample anyway.
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=1,random_state=seed)

    y = np.array(data["abd"])
    X = data.drop("abd",axis="columns")

    # Using as criterion mean squared error for a predictor in the log space is equivalent to minimize the mean absolute error in the natural space. Thus, at each tree-node split the algorithm is minimizing the mean absolute error of biomass density.
    cvs = cross_val_score(estimator,X,y,cv=rkf,scoring="neg_root_mean_squared_error")    
    # cvs = 0
    # estimator.fit(X,y)

    # Outputs for multi-objective optimization with NSGA-II with OpenMole.
    error = -1.0*np.mean(cvs)
    # print(error)

else:
    # Predicted value is the average log-transformed biomass density.
    y = np.array(data["abd"])
    avg = np.mean(y)
    error = np.sqrt(np.mean( (y - avg)**2 ))
    # print(error)

    





