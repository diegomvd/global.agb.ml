import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor
import pickle

class BGRBinaryEncoding(BaseEstimator, TransformerMixin):
    def __init__(self):
        self
    def fit(self, X , y=None):
        return self
    def transform(self, X):
        df = X.copy()

        df["bgr_tuple"] = [binary_encoding(t) for t in df.bgr]

        df_bin = df['bgr_tuple'].apply(pd.Series, index=['bgr1','bgr2','bgr3'])

        df = pd.concat([df,df_bin],axis='columns')

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
    return np.log(x+1)

def inverse_func(x):
    return np.exp(x)-1

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


def predictor_list(predictor_string):
    pred = predictor_string.split("-")
    return pred 


hp_file = "/home/dibepa/git/global.agb.ml/data/training/predictor_selection_onlybioclim/best_predictors_hp_absolute.csv"

# data_file = "/home/dibepa/git/global.agb.ml/data/training/tmp_preprocessed/tallo_learning_preprocessed_onlybioclim.csv"

data_file = "/home/dibepa/git/global.agb.ml/data/training/final_onlybioclim_with_notree/tallo_learning_preprocessed_onlybioclim_notrees.csv"

hp = pd.read_csv(hp_file)

hp["combination"] = hp.combination.apply(lambda x: predictor_list(x))

pipe_list = []

 # Iterate over each set of predictors and hyper-parameters
for index, row in hp.iterrows():

    bst = XGBRegressor(
        n_estimators=1000,
        learning_rate=float(row.e),
        max_depth = int(row.md),
        min_child_weight=float(row.mcw),
        subsample=float(row.subsample),
        min_split_loss = float(row.g),
        max_delta_step = float(row.mds),
        eval_metric = "rmse",
        objective='reg:squarederror',
    )

    regr = TransformedTargetRegressor(regressor=bst,func=func,inverse_func=inverse_func)

    plist = row.combination
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

    pipe_list.append(estimator)

estimators = [("pipe_{}".format(i),p) for i,p in enumerate(pipe_list)]

ensemble_regressor = VotingRegressor(
    estimators = estimators
)    


data_file = "/home/dibepa/git/global.agb.ml/data/training/detailed.allometries.model/ABD_training_dataset.csv"

data = pd.read_csv(data_file)
y = np.array(data["abd"])
X = data.drop("abd",axis="columns")

ensemble_regressor.fit(X,y)


pickle.dump(ensemble_regressor, open("/home/dibepa/git/global.agb.ml/data/training/detailed.allometries.model/abd_model_bioclim_notrees_noweights.pkl", "wb"))
# pickle.dump(ensemble_regressor, open("/home/dibepa/git/global.agb.ml/data/training/final_onlybioclim_with_notree/abd_model_onlybioclim_notrees.pkl", "wb"))
