import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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

    # For testing
    data = pd.read_csv("./training_data_2/tallo_learning_preprocessed_outcvloop_train_fold_1.csv")
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
    w=1.0

else:
    # Read data from the corresponding outer fold.
    data = pd.read_csv("tallo_learning_preprocessed_outcvloop_train_fold_5.csv")

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

# Extract the target, weights and predictors.
abd = np.array(data["abd"])#.to_numpy
weights = np.array(data["wsample"])#.to_numpy
predictor_list = build_predictor_list(predictor_dict)
predictors = np.array(data[predictor_list])

npredictors = number_of_predictors(predictor_list)

if npredictors > 0:

    cvs = np.zeros(10)
    if npredictors > 0.0: 
        # Set up the random forest. The only hyper-parameter is the minimum amount of samples in a leaf to control smoothing.
        rf = RandomForestRegressor(n_estimators=100)
        # Set up 4 folds for inner cross-validation with 10 repeats.
        rkf = RepeatedKFold(n_splits=4, n_repeats=1)
        
        # Define nearest neighbor imputer with default parameters to impute nan NDVI values in boreal forests.
        imputer = KNNImputer()
        estimator = make_pipeline(imputer, rf)

        # Cross-validate the random forest. 
        # Using as criterion mean squared error for a predictor in the log space is equivalent to minimize the mean absolute error in the natural space. Thus, at each tree-node split the algorithm is minimizing the mean absolute error of biomass density.
        if w == 1.0:
            cvs = cross_val_score(estimator,predictors,abd,cv=rkf,fit_params={"randomforestregressor__sample_weight": weights},scoring="neg_mean_squared_error")
        elif w == 0.0:
            cvs = cross_val_score(estimator,predictors,abd,cv=rkf,scoring="neg_mean_squared_error")    

    # Outputs for multi-objective optimization with NSGA-II with OpenMole.
    error = -1.0*np.mean(cvs)

else:
    # Predicted value is the average log-transformed biomass density.
    avg = np.mean(abd)
    error = np.mean( (abd - avg)**2 )



