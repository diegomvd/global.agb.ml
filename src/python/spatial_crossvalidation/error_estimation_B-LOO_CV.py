"""
Estimation of model error. We take model calibrated with random cross-validation to compare error estimation between Buffer-LOO CV and random CV. 
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline


def mean_absolute_error(ypred, y):
    return np.mean(np.abs(ypred-y))

def mean_signed_difference(ypred, y):
    return np.mean(ypred-y)

def mean_absolute_percentual_error(ypred, y):
    return 100 * np.mean(np.abs(ypred - y)/y)


dataset = pd.read_csv("")
predictors = []


indices = pd.read_csv("")
 
r0 = 100
r1 = 1500
dr = 100
radius_vector = np.arange(r0,r1,dr)

replicas = 10

df = pd.DataFrame()

for r in radius_vector:
    for id in indices.iterrows:
        
        train = dataset.drop(id.1)
        ytrain = train["abd"]
        Xtrain = train[predictors]

        test = dataset[id]
        ytest = test.abd
        Xpred = test[predictors]

        xgbr = XGBRegressor(
            n_estimators=1000,
            eval_metric = "rmse",
            objective='reg:squarederror',
            learning_rate =  0.02,
            max_depth =  5,
            min_child_weight = 1,
            subsample = 0.7,
            min_split_loss = 1,
            max_delta_step  = 1
        )

        imputer = KNNImputer()
        estimator = make_pipeline(imputer, xgbr)

        mae_list=[]
        msd_list=[]
        mape_list=[]
        for i in range(replicas):

            estimator.fit(Xtrain,ytrain)
            
            ypred = estimator.predict(Xpred)

            ypred_exp = np.exp(ypred)
            y_exp = np.exp(ytest)

            mae_list.append( mean_absolute_error(ypred_exp, y_exp))
            msd_list.append(mean_signed_difference(ypred_exp, y_exp))
            mape_list.append( mean_absolute_percentual_error(ypred_exp, y_exp))
        mae = np.mean(mae_list)    
        msd = np.mean(msd_list)
        mape = np.mean(mape_list)

        row = pd.DataFrame([  {"distance" : r, "id" : id, "mae" : mae, "msd" : msd , "mape" : mape }  ])

        df = pd.concat([df,row],axis=0, ignore_index=True)