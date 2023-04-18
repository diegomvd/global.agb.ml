import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline


def build_predictor_list(predictor_str):
    plist= (predictor_str).split("-")
    use_weights=False
    if "w" in plist: 
        use_weights = True
        plist.remove("w")
    if "bgr" in plist:
        plist.append("bgr1")
        plist.append("bgr2")
        plist.append("bgr3")     
        plist.remove("bgr")
    return plist, use_weights    

testing = True
if testing:

    # For testing
    data = pd.read_csv("./training_data_XGBR/tallo_learning_preprocessed_outcvloop_train_fold_1.csv")
    predictors = pd.read_csv("./training_data_XGBR/predictors.csv")
    pid = 38
    e = 1
    md = 3
    mcw = 1
    subsample = 0.8
    g = 0
    mds = 0 
    seed = 1

else:
    # Read data from the corresponding outer fold.
    data = pd.read_csv("tallo_learning_preprocessed_outcvloop_train_fold_1.csv")
    predictors = pd.read_csv("predictors.csv")

p = predictors[predictors["Unnamed: 0"] == pid]

npredictors = p.predictors.get(pid)
combination = p.combination.get(pid)

# Extract the target, weights and predictors.
y = np.array(data["abd"])
weights = np.array(data["wsample"])

predictor_list, use_weights = build_predictor_list(combination)
X = np.array(data[predictor_list])

cvs = np.zeros(10)

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

# Set up 4 folds for inner cross-validation with 1 repeat.
rkf = RepeatedKFold(n_splits=4, n_repeats=1)

# Define nearest neighbor imputer with default parameters to impute nan NDVI values in boreal forests.
imputer = KNNImputer()
estimator = make_pipeline(imputer, bst)
 
# Using as criterion mean squared error for a predictor in the log space is equivalent to minimize the mean absolute error in the natural space. Thus, at each tree-node split the algorithm is minimizing the mean absolute error of biomass density.
if use_weights:
    cvs = cross_val_score(estimator,X,y,cv=rkf,fit_params={"xgbregressor__sample_weight": weights},scoring="neg_root_mean_squared_error")
else:
    cvs = cross_val_score(estimator,X,y,cv=rkf,scoring="neg_root_mean_squared_error")    

# Outputs for multi-objective optimization with NSGA-II with OpenMole.
error = -1.0*np.mean(cvs)
print(error)





