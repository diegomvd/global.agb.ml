import numpy as np
import pandas as pd
from xgboost import XGBRegressor
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
        print(df)
        df["bgr_tuple"] = [binary_encoding(t) for t in df.bgr] 
        df["bgr1"] , df["bgr2"], df["bgr3"] = df.bgr_tuple.str
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

def mae(actual,predicted):
    return np.mean(np.abs(actual-predicted))  

def msd(actual,predicted):
    return np.mean(actual-predicted)  

def rmspe(actual,predicted):
    return np.sqrt(np.mean( ( (actual-predicted)/actual )**2) )

def mape(actual,predicted):
    return np.mean( np.abs( (actual-predicted)/actual ) )*100    

def mspd(actual,predicted):
    return np.mean( (actual-predicted)/actual )

# Best set of predictors and hyperparameters:
file = "/home/dibepa/git/global.agb.ml/data/training/predictor_selection_onlybioclim/best_predictors_hp_absolute.csv"
data_params = pd.read_csv(file)

def predictor_list(predictor_string):
    pred = predictor_string.split("-")
    return pred 

error_df = pd.DataFrame()

calculate_error_estimates = True

if calculate_error_estimates:

    # Iterate over each outer fold for train and test.
    for fold in data_params.fold.unique():

        data_train = pd.read_csv("/home/dibepa/git/global.agb.ml/data/training/preprocessed_onlybioclim/outcvloop_train_fold_{}.csv".format(fold))
        data_test = pd.read_csv("/home/dibepa/git/global.agb.ml/data/training/preprocessed_onlybioclim/outcvloop_test_fold_{}.csv".format(fold))

        y_train = np.array(data_train["abd"])
        y_test = np.array(data_test["abd"])

        # Transform the predictor combination to a list.
        params = data_params[data_params.fold == fold].reset_index(drop=True)

        plist = predictor_list(params.combination[0])

        # predictor_list = params.combination.apply(lambda x: predictor_list(x))[0]
        md = int(params.md)
        e = float(params.e)
        mcw = float(params.mcw)
        mds = float(params.mds) 
        g = float(params.g)
        subsample = float(params.subsample)    
                            

        repeats = 10
    
        bst = XGBRegressor(
                n_estimators=100,
                learning_rate=e,
                max_depth = md,
                min_child_weight=mcw,
                subsample=subsample,
                min_split_loss = g,
                max_delta_step = mds,
                eval_metric = "rmse",
                objective='reg:squarederror',
                )
        
        # Log-transform target observable.
        regr = TransformedTargetRegressor(regressor=bst,func=func,inverse_func=inverse_func)

        # Log-transform precipitation related variables and binarize biogeographic realm.
        log_preds = log_predictors(plist)
        if len(log_preds)>0:
            if "bgr" in plist:
                estimator = Pipeline([
                    ("col_extract", ColumnExtractor(plist)),
                    ("bgr_binary", BGRBinaryEncoding()),
                    ("log_water", LogarithmizeWaterObservables(plist)),
                    ("regressor",regr)
                ])
            else:   
                estimator = Pipeline([
                    ("col_extract", ColumnExtractor(plist)),
                    ("log_water", LogarithmizeWaterObservables(plist)),
                    ("regressor",regr)
                ])
        else:
            if "bgr" in plist:
                estimator = Pipeline([
                    ("col_extract", ColumnExtractor(plist)),
                    ("bgr_binary", BGRBinaryEncoding()),
                    ("regressor",regr)
                ])
            else:
                estimator = Pipeline([
                    ("col_extract", ColumnExtractor(plist)),
                    ("regressor",regr)
                ])
            
        X_train = data_train.drop("abd",axis="columns")
        X_test = data_test.drop("abd",axis="columns")

        mape_list = []
        mae_list = []
        msd_list = []

        for r in range(repeats): 

            estimator.fit(X_train,y_train)
            y_pred = estimator.predict(X_test)  

            mape_list.append(mape(y_test,y_pred))
            mae_list.append(mae(y_test,y_pred))
            msd_list.append(msd(y_test,y_pred))
        
        mape_mean = np.mean(mape_list)
        mae_mean = np.mean(mae_list)
        msd_mean = np.mean(msd_list)
        
        error_row = pd.DataFrame([{
            "foldId" : fold,
            "mape":mape_mean,
            "mae":mae_mean,
            "msd": msd_mean 
        }])

        error_df = pd.concat([error_df,error_row],axis=0, ignore_index=True)

    # print(error_df)
    # error_df.to_csv("/home/dibepa/git/global.agb.ml/data/training/predictor_selection_onlybioclim/crossvalidation_error_1500estimators.csv")
else:

    error_df = pd.read_csv("./training_data/innercvloop_optimization/crossvalidation_error.csv")
    mean_error_df = error_df.groupby("predictors").mean().drop(["foldId","Unnamed: 0"],axis = "columns").reset_index()
    print(mean_error_df)
            