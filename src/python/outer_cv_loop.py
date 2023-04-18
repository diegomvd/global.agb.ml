import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import scipy as sp
import os
import re

datadir = "./training_data_2/"

selected_predictors = pd.read_csv("./predictor_selection_2/best_predictors.csv")

repeats = 10

test_set = {}
train_set = {}
folds = []

# Each file is the last population of the GA used for predictor selection for each one of the 5 data folds.
for file in os.listdir(datadir):
    #print(file)
    foldId = int( file[file.index("_fold_")+len("_fold_")]  )
    folds.append(foldId)
    if "tallo_learning_preprocessed_outcvloop_test_fold_" in file:
        if datadir + file not in test_set:
            test_set[foldId] = datadir+file
        else:
            pass
    elif "tallo_learning_preprocessed_outcvloop_train_fold_" in file:
        if datadir + file not in train_set:
            train_set[foldId] = datadir+file
        else:
            pass         

def mae(actual,predicted):
    return np.mean(np.abs(actual-predicted))  

def msd(actual,predicted):
    return np.mean(actual-predicted)  

def rmspe(actual,predicted):
    return np.sqrt(np.mean( ( (actual-predicted)/actual )**2) )

def mape(actual,predicted):
    return np.mean( np.abs( (actual-predicted)/actual ) )    

def mspd(actual,predicted):
    return np.mean( (actual-predicted)/actual )


def get_predictors_combinations_columns_list(combinations):
    """
    List of list of column names.
    """
    # Extract different set of predictors as a List of List of Strings.
    predictors_list = []
    for comb in combinations:
        pred1 = re.findall('\-(.*?)\-', comb)
        pred2 = re.findall('\-(.*?)\-', "-"+comb+"-" )
        pred = pred1 + pred2
        if "bgr" in pred:
            pred.remove("bgr")
            pred.append("bgr1")
            pred.append("bgr2")
            pred.append("bgr3")
        predictors_list.append(pred)
    return predictors_list    



def deencode_bgr(data):
    # # Palearctic   = 0 0 0
    # # Indomalayan = 0 1 0
    # # Australasia = 0 0 1
    # # Nearctic    = 0 1 1
    # # Afrotropic  = 1 0 0
    # # Neotropic   = 1 1 0

    data["bgr"] = [""]*data.shape[0]
    data["bgr"] = np.where(
                (data["bgr1"] == 0) & (data["bgr2"] == 0) & (data["bgr3"] == 0),
                 "Palearctic", data["bgr"] )
    data["bgr"] = np.where(
                (data["bgr1"] == 0) & (data["bgr2"] == 1) & (data["bgr3"] == 0),
                 "Indomalayan", data["bgr"] )
    data["bgr"] = np.where(
                (data["bgr1"] == 0) & (data["bgr2"] == 0) & (data["bgr3"] == 1),
                 "Australasia", data["bgr"] )
    data["bgr"] = np.where(
                (data["bgr1"] == 0) & (data["bgr2"] == 1) & (data["bgr3"] == 1),
                 "Nearctic", data["bgr"] )
    data["bgr"] = np.where(
                (data["bgr1"] == 1) & (data["bgr2"] == 0) & (data["bgr3"] == 0),
                 "Afrotropic", data["bgr"] )
    data["bgr"] = np.where(
                (data["bgr1"] == 1) & (data["bgr2"] == 1) & (data["bgr3"] == 0),
                 "Neotropic", data["bgr"] )
    return data["bgr"]


logCorrection = False
error_estimation = pd.DataFrame()
prediction_vs_actual = pd.DataFrame()

for fid in np.unique(folds):

    # Load data for a given split.
    train = pd.read_csv(train_set[fid])
    test = pd.read_csv(test_set[fid])

    test_bgr = deencode_bgr(test)    

    # Filter data to get only the best combinations of predictors selected with a GA in the inner nested-CV loop.
    fold_data = selected_predictors[selected_predictors.fold == fid]

    # Iterate over all the possible number of predictors selected to estimate the Pareto front for each fold.
    npred = np.unique(fold_data.predictors)[:]
    for n in npred:

        # Get the most selected predictor combination for a given number of predictors.
        combs = fold_data[(fold_data.predictors == n)]
        combs = combs[(combs.samples == np.max(combs.samples)) ].combination

        predictors_list = get_predictors_combinations_columns_list(combs)

        use_weights = False
        print(predictors_list)
        if "w" in predictors_list[0]:
            use_weights = True
            print(predictors_list)
            predictors_list[0].remove("w")
            print(predictors_list)
        
        # Repeat evaluation of error for more robust estimation.
        # Lists for corrected predictions.
        mae_list = []
        msd_list = []
        mae_log_list = []
        msd_log_list = []
        mape_list = []

        for iter in range(repeats):

            # Final prediction is an ensemble of different RF built with different optimal sets of predictors.
            ensemble = pd.DataFrame()

            # Iterating over combinations is to produce ensemble predictions.
            for i, pred in enumerate(predictors_list):

                print(pred)
                # Create a Extreme Gradient Boost regressor.
                bst = XGBRegressor(
                    n_estimators=1000,
                    subsample=0.8,
                    max_depth = 5,
                    objective='reg:squarederror'
                )    

                # bst = XGBRegressor(
                #     n_estimators=150,
                #     learning_rate=e,
                #     max_depth = md,
                #     min_child_weight=mcw,
                #     subsample=0.8,
                #     random_state = seed,
                #     eval_metric = "rmse",
                #     objective='reg:squarederror',
                #     tree_method = "approx"
                #     #early_stopping_rounds=30
                # )    
               
                X_train = np.array(train[pred])
                y_train = np.array(train["abd"])
                
                if use_weights:
                    weights = np.array(train["wsample"])
                    # Fitted model.
                    regressor = bst.fit(X_train,y_train,sample_weight=weights)
                else:
                    regressor = bst.fit(X_train,y_train)    
                
                X_test = np.array(test[pred])

                y_log_pred = regressor.predict(X_test)
                y_pred = np.exp(y_log_pred)
                col = pd.DataFrame({
                    "pred_set":i, 
                    "log": y_log_pred, 
                    "backtransformed": y_pred 
                    })    
                ensemble = pd.concat([ensemble,col],axis=1)
                print(ensemble)
    
            #print("Prediction Average")
            if ensemble.shape[1]>3:
                prediction_avg_log = ensemble.log.mean(axis=1)
                #print(prediction_avg_log)
                prediction_avg_bt = ensemble.backtransformed.mean(axis=1)
                #print(prediction_avg_bt)
            else:
                prediction_avg_log = ensemble.log
                prediction_avg_bt = ensemble.backtransformed

            #print("Target")
            test_log_target = np.array(test["abd"])
            test_target = np.exp(test_log_target)
            #print(pd.DataFrame(test_target))

            mae_list.append( mae(test_target,prediction_avg_bt.to_numpy()))
            msd_list.append( msd(test_target,prediction_avg_bt.to_numpy()))
            mape_list.append( mape(test_target,prediction_avg_bt.to_numpy()))

            mae_log_list.append( mae(test_log_target,prediction_avg_log.to_numpy()) )
            msd_log_list.append( msd(test_log_target,prediction_avg_log.to_numpy()) )

            col_pred_vs_actual = pd.DataFrame(
                {
                    "fold" : [fid]*prediction_avg_bt.shape[0],
                    "predictors": [n]*prediction_avg_bt.shape[0],
                    "repeat" : [iter]*prediction_avg_bt.shape[0],
                    "predicted" : prediction_avg_bt,
                    "actual" : test_target,
                    "bgr" : test_bgr,
                    "tcov" : test["tcov"],
                    "wavai" : test["wavai"],
                    "ts" : test["ts"],
                    "ps" : test["ps"],
                    "ndviw" : test["ndviw"]
                }
            )
            prediction_vs_actual = pd.concat([prediction_vs_actual,col_pred_vs_actual],axis=0)
            # print(prediction_vs_actual)

            #print(mae_list)
            #print(msd_list)
            #print(mae_log_list)
            #print(msd_log_list)
          #  break
        #break 
    #break   
        avg_mae = np.mean(mae_list)
        avg_msd = np.mean(msd_list)
        avg_mape = np.mean(mape_list)
        avg_log_mae = np.mean(mae_log_list)
        avg_log_msd = np.mean(msd_log_list)

        row = pd.DataFrame([{
            "fold":fid,
            "predictors":n,
            "mae":avg_mae,
            "msd":avg_msd,
            "mape":avg_mape,
            "mae_log":avg_log_mae,
            "msd_log":avg_log_msd
        }])

        # row = pd.DataFrame([{
        #     "fold":fid,
        #     "predictors":n,
        #     "rmse":avg_rmspe,
        #     "rmspe_raw":avg_raw_rmspe
        #    # "mape": avg_mape,
        #    # "mape_raw":avg_raw_mape,
        #    # "mspd": avg_mspd,
        #    # "mspd_raw":avg_raw_mspd
        # }])
        error_estimation = pd.concat([error_estimation,row],axis=0)
        print(error_estimation)

prediction_vs_actual.to_csv("./training_results_3/predicted_vs_actual_bgr_2.csv")

error_estimation.to_csv("./training_results_3/error_by_fold_2.csv")

error_estimation.drop("fold",axis="columns").groupby("predictors").mean().to_csv("./training_results_3/global_error_2.csv")
        

            
# def log_correction_factor(actual, predicted_log, error, **kwargs):
#     """Determine correction delta for exp transformation"""
#     def cost_func(delta):
#         e = error( actual, np.exp(delta + predicted_log) )
#         return e 
#     res = sp.optimize.minimize(cost_func, 0., **kwargs)
#     if res.success:
#         return res.x
#     else:
#         raise RuntimeError(f"Finding correction term for exponential transformation failed!\n{res}")


# def correcting_factor_splits(train_data):
#     X_train = train_data.drop(["abd","wsample"],axis="columns")
#     print(X_train.shape)
#     X_train["wsample"] = train_data["wsample"] # Making sure wsample is the last column.
#     y_train = train_data["abd"]
#     X_train, X_val, y_train, y_val = train_test_split(X_train.to_numpy(), y_train.to_numpy())
#     weights = X_train[:,-1] # last column
#     X_train = X_train[:,:-1]
#     X_val = X_val[:,:-1]
#     return X_train, X_val, y_train, y_val, weights



# def calculate_correction_factors(y_val_log_pred,y_val):
#     corr_rmspe = log_correction_factor(y_val,y_val_log_pred,rmspe)
#     print(corr_rmspe)
#     #corr_mape = log_correction_factor(y_val,y_val_log_pred,mape,options={'gtol': 1e-02})
#     #corr_mspd = log_correction_factor(y_val,y_val_log_pred,mspd,options={'gtol': 0.5})
#     factors = {
#         "rmspe" : corr_rmspe
#        # "mape" : corr_mape,
#        # "mspd" : corr_mspd
#     }
#     return factors


# if logCorrection:
#                     cols = ["abd"] + pred + ["wsample"]
#                     # Training data for this predictor set.
#                     train_filtered = train[cols]

#                     # Create a validation split to find optimal correction factor: this needs correction.
#                     X_train, X_val, y_train, y_val, weights = correcting_factor_splits(train_filtered)

#                     # Fitted model.
#                     regressor = rf.fit(X_train,y_train,sample_weight=weights)
                    
#                     # Optimize the correction factor.
#                     y_val_log_pred = regressor.predict(X_val)
#                     corr_factors = calculate_correction_factors(y_val_log_pred,np.exp(y_val))

#                     # Test data for prediction with and without correction.
#                     X_test = np.array(test[pred])
                    
#                     # Predict the test dataset.
#                     y_log_pred = regressor.predict(X_test)

#                     # Transform and add correction terms.
#                     raw_prediction = np.exp(y_log_pred)
#                     corr_prediction_rmspe = np.exp(y_log_pred + corr_factors["rmspe"]) 
#                     #corr_prediction_mape = np.exp(y_log_pred + corr_factors["mape"])
#                     #corr_prediction_mspd = np.exp(y_log_pred + corr_factors["mspd"])
                
#                     # Store the predictions in the ensemble dataframe.
#                     col = pd.DataFrame({
#                         "pred_set":i, 
#                         "raw": raw_prediction, 
#                         "rmspe": corr_prediction_rmspe 
#                         #"mape": corr_prediction_mape, 
#                         #"mspd": corr_prediction_mspd
#                         })    
#                     ensemble = pd.concat([ensemble,col],axis=1)
#                     #print(ensemble)