import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Continuous


"""
TODO: fit_params cannot be passed to GASearchCV, thus all the fits realized here are without using instance weights.
"""

def mean_absolute_error(ypred, y):
    return np.mean(np.abs(ypred-y))

def mean_signed_difference(ypred, y):
    return np.mean(ypred-y)

def mean_absolute_percentual_error(ypred, y):
    return 100 * np.mean(np.abs(ypred - y)/y)

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

data = pd.read_csv("/home/dibepa/git/global-above-ground-biomass-ml/revised_model/sklearn-gen/training_data_XGBR/tallo_learning_preprocessed_outcvloop_train_fold_1.csv")
predictors = pd.read_csv("/home/dibepa/git/global-above-ground-biomass-ml/revised_model/sklearn-gen/training_data_XGBR/predictors.csv")

# Hyper-parameter tuning is done for every set of previously selected predictors to select the best ones.
df =pd.DataFrame()
for fold in [1,2,3,4,5]:
    data = pd.read_csv("/home/dibepa/git/global-above-ground-biomass-ml/revised_model/sklearn-gen/training_data_XGBR/tallo_learning_preprocessed_outcvloop_train_fold_{}.csv".format(fold))
    test = pd.read_csv("/home/dibepa/git/global-above-ground-biomass-ml/revised_model/sklearn-gen/training_data_XGBR/tallo_learning_preprocessed_outcvloop_test_fold_{}.csv".format(fold))
    for pid in np.unique(predictors["Unnamed: 0"]):
        p = predictors[predictors["Unnamed: 0"] == pid]

        npredictors = p.predictors.get(pid)
        combination = p.combination.get(pid)

        # Extract the target, weights and predictors.
        y = np.array(data["abd"])

        weights = np.array(data["wsample"])

        predictor_list, use_weights = build_predictor_list(combination)
        X = np.array(data[predictor_list])

        # Initialize Extreme Gradient Boosting Regressor.
        xgbr = XGBRegressor(
            n_estimators=1000,
            eval_metric = "rmse",
            objective='reg:squarederror',
            )

        # Define nearest neighbor imputer with default parameters to impute nan NDVI values in boreal forests.
        imputer = KNNImputer()
        estimator = make_pipeline(imputer, xgbr)

        # Define space exploration for the genetic algorithm.
        param_grid = {'xgbregressor__learning_rate': Continuous(0.01, 0.6),
                    'xgbregressor__max_depth': Integer(3, 6),
                    'xgbregressor__min_child_weight': Continuous(1.0,10.0),
                    'xgbregressor__subsample': Continuous(0.6, 1.0),
                    'xgbregressor__min_split_loss': Continuous(0.0,100.0),
                    'xgbregressor__max_delta_step': Continuous(0.0,10.0)
                    }


        # Set up 4 folds for inner cross-validation with 1 repeat.
        cv = KFold(n_splits=4, shuffle = True)

        # Cross-validation with evolutionary alogirthm.
        evolved_estimator = GASearchCV(estimator=estimator,
                                    cv=cv,
                                    scoring="neg_root_mean_squared_error",
                                    param_grid=param_grid,
                                    n_jobs=-1,
                                    verbose=True,
                                    population_size=10,
                                    generations=50
                                    )
        
        evolved_estimator.fit(X,y)

        # Getting best parameter set for this predictor set and this fold.
        best = evolved_estimator.best_params_
        best["predictors"] = pid
        best["fold"] = fold

        # Get evaluation metrics by predicting on the outer folds of the nested-CV.
        Xtest = np.array(test[predictor_list])
        ytest = np.array(test["abd"])
        ypred = evolved_estimator.predict(Xtest)

        ypred_exp = np.exp(ypred)
        y_exp = np.exp(ytest)

        best["mae"] = mean_absolute_error(ypred_exp, y_exp)
        best["msd"] = mean_signed_difference(ypred_exp, y_exp)
        best["mape"] = mean_absolute_percentual_error(ypred_exp, y_exp)
        
        df  = pd.concat([df,pd.DataFrame([best])], axis = 0 , ignore_index=True )
        print(df)

    df.to_csv("./xgbr_tuned_params_fold_{}.csv".format(fold),index=True)
df.to_csv("./xgbr_tuned_params_all.csv",index=True)    
 