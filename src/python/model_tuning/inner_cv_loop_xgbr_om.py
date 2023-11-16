import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline



def keep_predictor(plist,pname,pval):
    if pval>0.5:
        plist.append(pname)
    return plist


def build_predictor_list(predictor_dict):
    plist = []
    for predictor in predictor_dict:
        plist = keep_predictor(plist,predictor,predictor_dict[predictor])
    return plist    


def number_of_predictors(plist):
    npredictors = np.nan
    if "bgr1" in plist:
        npredictors = len(plist) - 2.0
    else:
        npredictors = len(plist)
    return npredictors   

testing = False

if testing: 
    fold = 1

if testing:

    # For testing
    data = pd.read_csv("/home/dibepa/git/global.agb.ml/data/training/preprocessed/outcvloop_train_fold_{}.csv".format(fold))
    yt = 0
    tdq = 0
    tcq = 0 
    twq = 0
    ts = 0
    ps = 0
    pet = 1
    ai = 0
    yp = 1
    pwm = 0
    tcov = 1
    bgr = 1
    ndvism = 1
    ndvif = 1
    ndviw = 1
    ndvisp = 1
    wavai = 0
    e = 1
    md = 3
    mcw = 1
    subsample = 0.8
    g = 0
    mds = 0 
    seed = 1

else:
    # Read data from the corresponding outer fold.
    data = pd.read_csv("outcvloop_train_fold_{}.csv".format(fold))


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
    "ai" : ai,
    "ps" : ps,
    "tcov" : tcov,
    "bgr1" : bgr,
    "bgr2" : bgr,
    "bgr3" : bgr,
    "ndvism" : ndvism,
    "ndvif" : ndvif,
    "ndviw" : ndviw,
    "ndvisp" : ndvisp,
    "wavai" : wavai
}

# Extract the target and predictors.
y = np.array(data["abd"])
predictor_list = build_predictor_list(predictor_dict)
X = np.array(data[predictor_list])

n_splits = 5
cvs = np.zeros(n_splits)

mdint = int(md)

bst = XGBRegressor(
        n_estimators=1000,
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

# Set up 5 folds for inner cross-validation with 1 repeat because the evolutionary algorithm will resample anyway.
rkf = RepeatedKFold(n_splits=n_splits, n_repeats=1,random_state=seed)

# Define nearest neighbor imputer with default parameters to impute nan NDVI values in boreal forests.
imputer = KNNImputer()
estimator = make_pipeline(imputer, bst)

print("Starting.")
 
# Using as criterion mean squared error for a predictor in the log space is equivalent to minimize the mean absolute error in the natural space. Thus, at each tree-node split the algorithm is minimizing the mean absolute error of biomass density.
cvs = cross_val_score(estimator,X,y,cv=rkf,scoring="neg_root_mean_squared_error")    

# Outputs for multi-objective optimization with NSGA-II with OpenMole.
error = -1.0*np.mean(cvs)
print(error)





